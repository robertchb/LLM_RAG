import math
import os.path
import random
from dataclasses import dataclass
import torch
import numpy as np
import datasets
from pprint import pprint
from torch.utils.data import Dataset
# 批处理数据时自动添加填充，以处理长度不一的输入序列。
from transformers import DataCollatorWithPadding
# 导入 PyTorch 的分布式通信包 torch.distributed 并命名为 dist，
# 这个包提供了跨多个进程进行通信的功能，支持分布式训练。
import torch.distributed as dist

from .arguments import DataArguments


class SameDatasetTrainDataset(Dataset):
    """Dataset to yield a batch of data at one time. All samples in the same batch comes from the same task.
      生成一个批次的数据，其中同一个批次中的所有样本都来自同一个任务。
    """
    def __init__(self, args: DataArguments, # 训练数据相关的参数和设置
                 batch_size: int,
                 seed: int,
                 process_index: int=0, # 当前进程的索引
                 num_processes: int=1 # 总共的进程数
                 ):
        train_datasets = []
        each_data_inxs = [] # 索引
        batch_size_inxs = [] # 批次大小
        pqloss_flag = [] # 特定的标记
        cur_all_num = 0
        # 获取的阈值，用于处理小数据集和决定何时丢弃合并后的小数据集。
        SMALL_THRESHOLD = args.small_threshold
        DROP_THRESHOLD = args.drop_threshold

        # 定义了不使用知识蒸馏和使用知识蒸馏时的数据特征结构。
        # 这些特征包括查询字符串、正样本序列、负样本序列，以及可选的正负样本得分。
        context_feat = datasets.Features({
            'query': datasets.Value('string'),
            'pos': datasets.Sequence(datasets.Value('string')),
            'neg': datasets.Sequence(datasets.Value('string'))
        })
        context_feat_kd = datasets.Features({
            'query': datasets.Value('string'),
            'pos': datasets.Sequence(datasets.Value('string')),
            'neg': datasets.Sequence(datasets.Value('string')),
            'pos_scores': datasets.Sequence(datasets.Value('float')),
            'neg_scores': datasets.Sequence(datasets.Value('float')),
        })

        # 确保 args.train_data 是一个列表且至少包含一个元素
        assert isinstance(args.train_data, list) and len(args.train_data) >= 1

        # 检查当前进程是否是主进程
        if dist.get_rank() == 0:
            self.print_batch_size(batch_size=batch_size, train_group_size=args.train_group_size)

        # 遍历列表中的每个数据目录
        for data_dir in args.train_data:
            # 检查 data_dir 是否是一个目录
            if not os.path.isdir(data_dir):
                raise FileNotFoundError(f"{data_dir} is a file, not a directionary")
            
            small_datasets = [] # 存储小数据集
            small_batch_size = math.inf # 一个无穷大的值
            
            # Add `parallel_` in `data_dir` to indicate that this dataset is parallel corpus
            # 检查该数据集是否为并行语料库
            flag = 'parallel_' in data_dir
            # 遍历数据目录中的每个文件
            for file in os.listdir(data_dir):
                if not (file.endswith('.json') or file.endswith('.jsonl')):
                    # 如果不是，则跳过该文件。
                    continue
                # 获取文件的完整路径
                file_path = os.path.join(data_dir, file)
                # 如果当前进程是主进程，则打印正在加载的数据文件路径。
                if dist.get_rank() == 0:
                    print(f'loading data from {file_path} ...')
                try:
                    # 使用指定的特征结构 context_feat 加载数据文件
                    temp_dataset = datasets.load_dataset('json', data_files=file_path, split='train', cache_dir=args.cache_path, features=context_feat)
                except:
                    # 加载一个包含知识蒸馏得分（pos_scores, neg_scores）的数据集
                    temp_dataset = datasets.load_dataset('json', data_files=file_path, split='train', cache_dir=args.cache_path, features=context_feat_kd)
                    # 如果不使用知识蒸馏，那么将从数据集中移除 pos_scores 和 neg_scores 这两列。
                    # 这是为了保持数据结构的一致性，即使在不需要这些得分的情况下也能正常加载数据。
                    if not args.knowledge_distillation:
                        temp_dataset = temp_dataset.remove_columns(['pos_scores', 'neg_scores'])
                # 检查加载的数据集 temp_dataset 是否为空
                if len(temp_dataset) == 0:
                    continue
                elif len(temp_dataset) < SMALL_THRESHOLD:
                    # 如果数据集的大小小于设定的小数据集阈值 SMALL_THRESHOLD
                    # 将这个小数据集添加到 small_datasets 列表中
                    small_datasets.append(temp_dataset)
                    # 更新 small_batch_size 的值为当前 small_batch_size 与通过 get_file_batch_size 方法计算得到的批次大小的较小值。
                    small_batch_size = min(small_batch_size, self.get_file_batch_size(file, batch_size, train_group_size=args.train_group_size))
                else:
                    # 如果设置了每个数据集的最大示例数且当前数据集的大小超过了这个限制，则从数据集中随机选择指定数量的样本构成新的数据集。
                    if args.max_example_num_per_dataset is not None and len(temp_dataset) > args.max_example_num_per_dataset:
                        temp_dataset = temp_dataset.select(
                            random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                    # 将处理后的数据集（可能已被截断）添加到 train_datasets 列表中
                    train_datasets.append(temp_dataset)
                    # 为当前数据集生成一个索引数组，然后加上到目前为止已累计的所有数据集的总样本数 (cur_all_num)，
                    # 并将这个数组添加到 each_data_inxs 列表中。
                    each_data_inxs.append(np.arange(len(temp_dataset)) + cur_all_num)
                    # 更新到目前为止的总样本数，加上当前数据集的大小。
                    cur_all_num += len(temp_dataset)
                    # 计算当前文件的批次大小并添加到 batch_size_inxs 列表中
                    batch_size_inxs.append(self.get_file_batch_size(file, batch_size, train_group_size=args.train_group_size))
                    pqloss_flag.append(flag)

            # 查 small_datasets 列表是否非空，确保有小数据集需要被处理。
            if len(small_datasets) > 0:
                # concatenate_datasets 函数将所有小数据集合并为一个大的数据集 small_dataset。
                small_dataset = datasets.concatenate_datasets(small_datasets)
                # 检查合并后的数据集大小是否大于或等于设定的丢弃阈值，这个阈值用来判断合并后的数据集是否足够大，
                # 以避免训练时使用过小的数据集可能导致的问题。
                if len(small_dataset) >= DROP_THRESHOLD:
                    # 将合并后的数据集添加到 train_datasets 列表中
                    train_datasets.append(small_dataset)
                    # 为合并后的数据集生成一个索引数组，并加上当前的总样本数 (cur_all_num)，
                    # 然后将这个索引数组添加到 each_data_inxs 列表中。
                    each_data_inxs.append(np.arange(len(small_dataset)) + cur_all_num)
                    # 更新总样本数，将合并后的数据集的样本数加到总数上。
                    cur_all_num += len(small_dataset)
                    # 将之前计算得到的小数据集的最小批次大小 (small_batch_size) 添加到 batch_size_inxs 列表中。
                    batch_size_inxs.append(small_batch_size)
                    pqloss_flag.append(flag)

        # 将所有已处理的训练数据集 train_datasets 合并为一个大的数据集
        self.dataset = datasets.concatenate_datasets(train_datasets)
        # 将 each_data_inxs 列表（包含每个数据集的索引数组）赋值给 self.each_data_inxs 属性
        self.each_data_inxs = each_data_inxs
        # 生成一个从 0 到 each_data_inxs 列表长度的数组，这个数组代表了每个数据集的索引，用于训练时选择数据集。
        self.datasets_inxs = np.arange(len(each_data_inxs))
        self.batch_size_inxs = batch_size_inxs
        # 将 pqloss_flag 列表（标记每个数据集是否为并行语料库）赋值给 self.pqloss_flag 属性
        self.pqloss_flag = pqloss_flag
        # 当前进程的索引
        self.process_index = process_index
        # 表示参与训练的总进程数
        self.num_processes = num_processes
        self.args = args
        # 这个比例参数用于决定数据的混洗程度
        self.shuffle_ratio = args.shuffle_ratio
        # 创建一个确定性的随机数生成器
        self.deterministic_generator = np.random.default_rng(seed)
        self.step = 0
        self.refresh_epoch()
    
    def print_batch_size(self, batch_size: int, # 原始批次大小
                         train_group_size: int): # 训练组大小
        # 每个字符串表示一个数据长度的范围，用于模拟不同长度的数据文件。
        length_list = ['0-500', '500-1000', '1000-2000', '2000-3000', '3000-4000', '4000-5000', '5000-6000', '6000-7000', '7000-inf']
        batch_size_dict = {
            k: self.get_file_batch_size(f"len-{k}.jsonl", batch_size, train_group_size) for k in length_list
        }
        # 每个字符串包含一个长度范围和对应的批次大小，用于打印显示。
        batch_size_list = [
            f'{length}: {batch_size_dict[length]}' for length in length_list
        ]
        print("=========================")
        print("Batch Size Dict:")
        pprint(batch_size_list)
        print("=========================")
    
    @staticmethod
    def get_file_batch_size(file: str, batch_size: int, train_group_size: int):
        '''这个方法假定不同长度范围的数据文件对内存和计算资源的需求不同，因此需要调整批次大小以适应不同的资源限制。'''
        if train_group_size == 8:
            # 80GB
            # 对于不同的数据长度范围，方法设定了不同的批次大小，以适应不同长度数据处理时的资源需求。
            if 'len-0-500.jsonl' in file:
                return 48
            elif 'len-500-1000.jsonl' in file:
                return 32
            elif 'len-1000-2000.jsonl' in file:
                return 20
            elif 'len-2000-3000.jsonl' in file:
                return 18
            elif 'len-3000-4000.jsonl' in file:
                return 14
            elif 'len-4000-5000.jsonl' in file:
                return 14
            elif 'len-5000-6000.jsonl' in file:
                return 12
            elif 'len-6000-7000.jsonl' in file:
                return 10
            elif 'len-7000-inf.jsonl' in file:
                return 8
            else:
                return batch_size
        elif train_group_size == 1:
            # 80GB
            if 'len-0-500.jsonl' in file:
                return 700
            elif 'len-500-1000.jsonl' in file:
                return 570
            elif 'len-1000-2000.jsonl' in file:
                return 388
            elif 'len-2000-3000.jsonl' in file:
                return 288
            elif 'len-3000-4000.jsonl' in file:
                return 224
            elif 'len-4000-5000.jsonl' in file:
                return 180
            elif 'len-5000-6000.jsonl' in file:
                return 157
            elif 'len-6000-7000.jsonl' in file:
                return 128
            elif 'len-7000-inf.jsonl' in file:
                return 104
            else:
                return batch_size
        else:
            return batch_size
    
    def refresh_epoch(self):
        '''用于每个新的训练周期开始时刷新或重置数据集的状态
        这个方法主要负责打乱数据集的顺序，动态调整批次大小，并准备好每个批次的数据。'''
        print(f'---------------------------*Rank {self.process_index}: refresh data---------------------------')
        # 随机打乱数据集索引(self.datasets_inxs)的顺序
        self.deterministic_generator.shuffle(self.datasets_inxs)
        # Dynamically adjust batch size
        # 存储准备好的批次数据
        batch_datas = []
        # 遍历打乱后的数据集索引
        for dataset_inx in self.datasets_inxs:
            # 对每个数据集的样本索引再次进行打乱，以确保数据的随机性。
            self.deterministic_generator.shuffle(self.each_data_inxs[dataset_inx])
            # 根据当前数据集的批次大小和进程数计算实际使用的批次大小
            cur_batch_size = self.batch_size_inxs[dataset_inx]*self.num_processes
            # 检查当前数据集是否被标记为并行语料库
            flag = self.pqloss_flag[dataset_inx]
            # 使用一个循环，按照计算出的批次大小(cur_batch_size)从当前数据集中提取样本，直到不能再形成一个完整的批次。
            for start_index in range(0, len(self.each_data_inxs[dataset_inx]), cur_batch_size):
                # 如果发现剩余样本数不足以形成两倍进程数(2 * self.num_processes)的批次，
                # 就跳过这部分样本，以保证每个批次的数据量足够且均衡。
                if len(self.each_data_inxs[dataset_inx]) - start_index < 2 * self.num_processes:
                    break
                # 将每个批次的数据索引和并行语料库的标志(flag)作为一个元组添加到 batch_datas 列表中
                batch_datas.append((self.each_data_inxs[dataset_inx][start_index:start_index+cur_batch_size], flag))
        # 打乱所有准备好的批次数据
        self.deterministic_generator.shuffle(batch_datas)
        self.batch_datas = batch_datas
        # 表示新的训练周期开始，步骤计数从头开始。
        self.step = 0

    def __getitem__(self, _):  
        batch_indices, pqloss_flag = self.batch_datas[self.step]
        cur_batch_size = int(len(batch_indices) / self.num_processes)
        batch_indices = batch_indices[self.process_index * cur_batch_size: (self.process_index + 1) * cur_batch_size]
        batch_data = self.dataset[batch_indices]
        self.step += 1
        queries, passages, teacher_scores = self.create_batch_data(batch_raw_data=batch_data)
        # print('rank, step, flag, query, passage:', dist.get_rank(), self.step, pqloss_flag, queries, passages)
        return queries, passages, teacher_scores, pqloss_flag

    def shuffle_text(self, text):
        '''根据设定的文本混洗比例(shuffle_ratio)，随机地对给定的文本(text)进行部分混洗。'''
        # 检查是否满足混洗条件，self.shuffle_ratio > 0：确保设置了正的混洗比例。
        # len(text) > 100：仅对长度超过 100 个字符的文本进行混洗，避免对太短的文本进行不必要的或过度的混洗。
        # random.random() < self.shuffle_ratio：通过随机概率决定是否对当前文本进行混洗，混洗发生的概率由 shuffle_ratio 控制。
        if self.shuffle_ratio > 0 and len(text) > 100 and random.random() < self.shuffle_ratio:
            # 用于存储被分割的文本块
            split_text = []
            # 将文本分成大约三等分的块，每块的大小由 chunk_size 决定。这里通过总长度除以 3 并向上取整来计算每块的大致大小。
            chunk_size = len(text)//3 + 1
            # 遍历文本长度
            for i in range(0, len(text), chunk_size):
                # 将文本按块切分并添加到 split_text 列表中
                split_text.append(text[i:i+chunk_size])
            # 对分割后的文本块进行随机混洗
            random.shuffle(split_text)
            # 将混洗后的文本块重新拼接成一个字符串，并返回这个混洗后的文本。
            return " ".join(split_text)
        else:
            # 如果不满足混洗条件，则直接返回原始文本。
            return text

    def create_batch_data(self, batch_raw_data):
        '''从原始批次数据(batch_raw_data)中提取查询(queries)、文段(passages)以及教师评分(teacher_scores)，
           并对正样本文段应用文本混洗策略。'''
        queries, passages = [], []
        teacher_scores = []
        # 遍历查询
        for i in range(len(batch_raw_data['query'])):
            # 将查询直接添加到queries列表
            queries.append(batch_raw_data['query'][i])
            # 从正样本(pos)中随机选择一个索引pos_inx，并对选中的正样本文段应用shuffle_text方法进行可能的文本混洗，
            # 然后添加到passages列表。
            pos_inx = random.choice(list(range(len(batch_raw_data['pos'][i]))))
            passages.append(self.shuffle_text(batch_raw_data['pos'][i][pos_inx]))
            # 如果提供了正样本评分('pos_scores')，则将选中正样本的评分添加到teacher_scores。
            if 'pos_scores' in batch_raw_data and batch_raw_data['pos_scores'][i] is not None:
                teacher_scores.append(batch_raw_data['pos_scores'][i][pos_inx])

            # 处理负样本，创建负样本索引集合
            neg_inx_set = list(range(len(batch_raw_data['neg'][i])))
            # 如果负样本数量少于训练组大小减一(self.args.train_group_size - 1)，
            # 通过重复负样本索引集合并随机抽取以确保所需数量的负样本索引。
            if len(batch_raw_data['neg'][i]) < self.args.train_group_size - 1:
                num = math.ceil((self.args.train_group_size - 1) / len(batch_raw_data['neg'][i]))
                neg_inxs = random.sample(neg_inx_set * num, self.args.train_group_size - 1)
            else:
                # 如果负样本数量充足，则直接从负样本索引集合中随机抽取所需数量的负样本索引。
                neg_inxs = random.sample(neg_inx_set, self.args.train_group_size - 1)

            # 如果存在负样本评分（'neg_scores'），对每个选中的负样本索引和对应评分进行排序，并根据评分的降序排列。
            if 'neg_scores' in batch_raw_data and batch_raw_data['neg_scores'][i] is not None:
                neg_scores = [(x, batch_raw_data['neg_scores'][i][x]) for x in neg_inxs]
                neg_scores = sorted(neg_scores, key=lambda x:x[1], reverse=True)
                neg_inxs = [x[0] for x in neg_scores]
                teacher_scores.extend([x[1] for x in neg_scores])
            # 根据排序后的负样本索引集合（neg_inxs），从原始数据中提取对应的负样本文段，并将这些文段追加到passages列表。
            negs = [batch_raw_data['neg'][i][x] for x in neg_inxs]
            passages.extend(negs)

            # 检查teacher_scores与passages列表长度匹配
            if len(teacher_scores) > 0 and len(passages) > 0:
                # 确保每个文段都有一个对应的教师评分
                assert len(teacher_scores) == len(passages)
        # 如果设置了查询指令（query_instruction_for_retrieval），则在每个查询前添加该指令。
        if self.args.query_instruction_for_retrieval is not None:
            queries = [self.args.query_instruction_for_retrieval+q for q in queries]
        # 如果设置了文段指令（passage_instruction_for_retrieval），则在每个文段前添加该指令。
        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
        
        if len(teacher_scores) == 0:
            teacher_scores = None

        # 返回包含处理后的查询（queries）、文段（passages）以及可选的教师评分（teacher_scores）的元组。
        return queries, passages, teacher_scores
    
    def __len__(self):
        '''返回数据集的长度，这个长度是基于批次数据 self.batch_datas 的数量和训练涉及的总进程数 self.num_processes 计算得到的。'''
        return len(self.batch_datas) * self.num_processes


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    用于将一系列特征（通常是模型输入的元组列表）转换成独立的查询和文段列表，
    同时对这些数据进行适当的处理，如填充等，以便它们可以被模型正确处理。
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features):
        # 从 features 中提取查询和文段数据
        query = [f[0] for f in features]
        passage = [f[1] for f in features]
        
        teacher_scores = None
        # 检查 features 中的元组是否包含超过两个元素（即是否存在教师评分）
        if len(features[0]) > 2:
            # 如果存在，它将提取教师评分到 teacher_scores 列表中。
            teacher_scores = [f[2] for f in features]
            if teacher_scores[0] is None:
                # 如果提取的第一个教师评分是 None，则保持 teacher_scores 为 None。
                teacher_scores = None
            else:
                # 否则，将 teacher_scores 列表转换为 PyTorch 的 FloatTensor
                teacher_scores = torch.FloatTensor(teacher_scores)
        
        flag = None
        # 检查features中的元组是否包含四个元素
        if len(features[0]) == 4:
            # 除了查询、文段和教师评分外，还有一个额外的标志（可能用于指示某种状态或属性）
            flag = [f[3] for f in features][0]

        # 展平查询和文段列表
        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        # 分词处理
        q_collated = self.tokenizer(
            query,
            # padding='max_length',     # used for adjusting the batch size in `get_file_batch_size()`
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            # padding='max_length',     # used for adjusting the batch size in `get_file_batch_size()`
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )

        # 如果存在teacher_scores，它们被调整为与查询输入张量（q_collated['input_ids']）的长度相匹配的形状。
        if teacher_scores is not None:
            teacher_scores = teacher_scores.reshape((len(q_collated['input_ids']), -1))

        # 返回一个包含处理后的查询（"query": q_collated）、文段（"passage": d_collated）、
        # 教师评分（"teacher_scores": teacher_scores）和双向标志（"bi_directions": flag）的字典。
        return {"query": q_collated, "passage": d_collated, "teacher_scores": teacher_scores, "bi_directions": flag}
