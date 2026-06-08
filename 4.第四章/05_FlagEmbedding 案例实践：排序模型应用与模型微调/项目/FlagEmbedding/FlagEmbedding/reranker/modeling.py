import logging

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel, TrainingArguments
# SequenceClassifierOutput类是序列分类任务的模型输出的标准格式，通常包括预测结果、损失值等信息。
from transformers.modeling_outputs import SequenceClassifierOutput

from .arguments import ModelArguments, DataArguments
# 创建了一个日志记录器
logger = logging.getLogger(__name__)


class CrossEncoder(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, # 预训练模型对象
                 model_args: ModelArguments, # 自定义的模型参数类
                 data_args: DataArguments, # 自定义的数据处理参数类
                 train_args: TrainingArguments # 包含训练过程中的参数
                 ):
        super().__init__()
        self.hf_model = hf_model
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        self.config = self.hf_model.config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        # 在这里，它注册了一个名为target_label的缓冲区，用零张量初始化，并根据self.train_args.per_device_train_batch_size设置其大小。
        # 这个张量的数据类型是torch.long。注册的缓冲区不会被视为模型参数，但它们可以在模型保存和加载时与模型一起被保存和加载。
        self.register_buffer(
            'target_label',
            torch.zeros(self.train_args.per_device_train_batch_size, dtype=torch.long)
        )

    def gradient_checkpointing_enable(self, **kwargs):
        '''梯度检查点是一种内存优化技术，它可以在训练大型模型时减少内存的消耗，但会略微增加计算时间。'''
        self.hf_model.gradient_checkpointing_enable(**kwargs)

    def forward(self, batch):
        # 初始化的预训练模型，传入一个批次的数据batch。
        ranker_out: SequenceClassifierOutput = self.hf_model(**batch, return_dict=True)
        # 从模型输出中提取逻辑回归（logits）值，这些值是模型对每个类别的原始预测得分，通常在应用softmax函数之前的值。
        logits = ranker_out.logits
        # 否处于训练模式
        # self.training是nn.Module的一个属性，当调用model.train()时会被设置为True，调用model.eval()时会被设置为False。
        if self.training:
            # 将逻辑回归（logits）值重塑为指定的形状，以便进行分组损失计算。
            # 这里的形状由批量大小（每个设备上的训练批量大小）和组大小（训练组大小）决定。
            scores = logits.view(
                self.train_args.per_device_train_batch_size,
                self.data_args.train_group_size
            )
            # 计算交叉熵损失。这里的self.target_label是之前注册的缓冲区，它被初始化为全零，用于计算与scores的损失。
            loss = self.cross_entropy(scores, self.target_label)

            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            # 如果不是训练模式，直接返回ranker_out。在评估或推理模式下，通常不需要计算损失，只关心模型的预测结果。
            return ranker_out

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        # 加载一个预训练的序列分类模型
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        # 创建CrossEncoder类的实例，将加载的预训练模型和其他参数传递给CrossEncoder的构造函数。
        reranker = cls(hf_model, model_args, data_args, train_args)
        # 返回创建好的CrossEncoder实例
        return reranker

    def save_pretrained(self, output_dir: str):
        '''将CrossEncoder中使用的Hugging Face模型及其权重保存到指定的目录'''
        # 获取模型的状态字典，其中包含了模型所有参数的名称和对应的权重。
        state_dict = self.hf_model.state_dict()
        # 创建了一个新的状态字典，遍历原始状态字典中的每一项，将权重数据移动到CPU上，并克隆一份。
        # 这是因为在保存模型时，最好确保模型的权重不依赖于特定的设备（如GPU），这样加载模型时就不会遇到设备不兼容的问题。
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        # 将模型和处理后的状态字典保存到output_dir指定的目录中
        self.hf_model.save_pretrained(output_dir, state_dict=state_dict)
