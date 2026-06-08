import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import datasets
import torch
from torch.utils.data import Dataset
# DataCollatorWithPadding类用于动态地对批处理中的样本进行填充，以保证批处理中的所有样本具有相同的长度
from transformers import DataCollatorWithPadding
#  BatchEncoding是处理后的批数据的封装，通常包含输入给模型的张量。
from transformers import PreTrainedTokenizer, BatchEncoding

from .arguments import DataArguments


class TrainDatasetForCE(Dataset):
    '''用于加载和准备训练数据'''
    def __init__(
            self,
            args: DataArguments, # 训练数据相关的配置
            tokenizer: PreTrainedTokenizer,
    ):
        # 检查args.train_data指定的路径是否为一个目录
        if os.path.isdir(args.train_data):
            # 如果是，意味着训练数据分散在该目录下的多个文件中。
            train_datasets = []
            # 遍历每个文件
            for file in os.listdir(args.train_data):
                # 指定只加载训练集部分
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                train_datasets.append(temp_dataset)
            # 如果训练数据分散在多个文件中，使用datasets.concatenate_datasets函数将它们合并为一个大的数据集。
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            #  如果args.train_data指定的是一个文件而不是目录，直接加载这个文件作为数据集。
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        # 数据集的总长度
        self.total_len = len(self.dataset)

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        '''单个样本的创建
        qry_encoding: 查询内容
        doc_encoding: 文档内容
        '''
        # 对这两段文本进行编码
        item = self.tokenizer.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False, # 不对序列进行自动填充到最大长度
        )
        return item

    def __len__(self):
        '''返回数据集的总长度'''
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        '''从数据集中构造出包含一条查询(query)和多个文档（一个正例pos和若干负例neg）的批次数据。'''
        # 从数据集中获取索引对应的查询文本
        query = self.dataset[item]['query']
        # 随机选择一个与当前查询相关的正例文档
        pos = random.choice(self.dataset[item]['pos'])
        # 如果负例文档的数量少于所需的数量（由self.args.train_group_size - 1确定）
        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            # 如果可用的负样本不足，计算需要重复使用现有负样本的次数。
            # 这里使用math.ceil函数确保重复的次数是足够的，即使在不整除的情况下也能保证有足够的负样本。
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            # 将负样本列表通过乘以num重复扩展，然后使用random.sample从扩展后的列表中随机选取所需数量的负样本。
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        # 存储批次数据
        batch_data = []
        # 将查询和选中的正例文档编码后加入到批次数据列表
        batch_data.append(self.create_one_example(query, pos))
        # 对于每个选中的负例文档，使用create_one_example方法进行编码，并将结果加入到批次数据列表。
        for neg in negs:
            batch_data.append(self.create_one_example(query, neg))
        # 返回构造好的批次数据列表
        return batch_data

'''举例子讲解在负样本不足的情况下，如何构造负样本？

假设在一个训练任务中，我们希望每个训练组包含1个正样本和7个负样本，即 train_group_size = 8
然而，对于某个查询query，我们在数据集中仅找到了2个负样本。我们将如何处理这种情况，以确保每个训练组能够包含足够的负样本呢？
下面通过这个假设的例子来解释上述代码的计算过程：
1. 设定期望的训练组大小:
    - train_group_size = 8
    - 这意味着每个训练组需要1个正样本和7个负样本。

2. 查看当前查询对应的负样本数量:
    - 假设找到的负样本数量为2，即len(neg) = 2。

3. 检查负样本是否足够:
    - 需要的负样本数量为train_group_size - 1 = 7。
    - 实际的负样本数量为2，所以不足。

4. 计算需要重复使用负样本的次数:
    - 用math.ceil((train_group_size - 1) / len(neg))计算重复次数。
    - 计算得到math.ceil(7 / 2) = 4。
    - 这意味着我们需要将现有的负样本列表重复4次，以确保有足够的负样本可以选择。

5. 构造足够数量的负样本:
    - 将负样本列表重复4次，假设原始负样本为[neg1, neg2]，则重复后为[neg1, neg2, neg1, neg2, neg1, neg2, neg1, neg2]。
    - 然后从这个扩展的列表中随机选择7个负样本。

通过这个过程，即使原始的负样本数量不足，我们也能通过重复使用现有负样本的方式来补足数量，从而满足训练组中负样本的需求。
这种方法保证了训练数据的一致性，即每个训练组都包含了指定数量的正样本和负样本，对于一些需要大量负样本参与的排序或检索任务来说尤为重要。

'''


# DataCollatorWithPadding是transformers库中的一个类，用于处理批量数据的填充。
@dataclass
class GroupCollator(DataCollatorWithPadding):
    def __call__(
            self, features # 这是一个列表，其中的每个元素代表一个训练样本的特征。
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # 检查features列表的第一个元素是否为列表
        if isinstance(features[0], list):
            # 如果features的第一个元素是列表，将所有这些列表的元素“展平”成一个单一的列表。
            features = sum(features, [])
        return super().__call__(features)
