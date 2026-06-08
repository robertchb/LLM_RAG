import logging
import os
from typing import Optional

import torch
from transformers.trainer import Trainer

from .modeling import CrossEncoder

logger = logging.getLogger(__name__)


class CETrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        '''用于保存模型和相关配置到指定的目录'''
        # 保存模型的目录
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        # 如果目录不存在，makedirs会创建它。
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # 首先检查模型对象（self.model）是否有save_pretrained方法。
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(f'MODEL {self.model.__class__.__name__} ' f'does not support save_pretrained interface')
        else:
            # 保存模型
            self.model.save_pretrained(output_dir)
        # 如果分词器存在，并且当前进程是主进程
        if self.tokenizer is not None and self.is_world_process_zero():
            # 保存分词器
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        # 将训练参数（self.args）保存为二进制文件，文件名为"training_args.bin"
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model: CrossEncoder, inputs):
        '''用于计算损失'''
        # 从返回的字典中取出损失值
        return model(inputs)['loss']
