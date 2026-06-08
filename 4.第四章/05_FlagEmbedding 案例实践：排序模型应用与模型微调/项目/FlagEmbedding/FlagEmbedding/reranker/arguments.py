import os
# dataclass装饰器自动为类添加特殊方法，如__init__()、__repr__()等。
# field函数用于自定义数据类的字段属性
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    # 预训练模型的名称或路径
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # 预训练配置的名称或路径
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    # 分词器的名称或路径
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    # 下载的预训练模型存储在本地的目录
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataArguments:
    '''用于管理和存储与训练数据相关的参数'''
    # 训练数据的路径
    train_data: str = field(
        default=None, metadata={"help": "Path to corpus"}
    )
    # 训练时的组大小
    train_group_size: int = field(default=8)
    # 输入文本的最大序列长度
    max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for input text. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    def __post_init__(self):
        '''这个方法会在数据类的自动生成的__init__方法执行完毕后被调用'''
        # 检查train_data指定的路径是否存在
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")
