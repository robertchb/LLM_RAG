import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, TrainingArguments
from transformers import (
    HfArgumentParser,
    set_seed,
)

from .arguments import ModelArguments, DataArguments
from .data import TrainDatasetForCE, GroupCollator
from .modeling import CrossEncoder
from .trainer import CETrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # 将命令行参数解析为上述三个数据类的实例。
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        # 抛出一个ValueError异常，提示用户输出目录已存在且不为空，除非使用--overwrite_output_dir选项。
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    # 配置了日志的格式、日期格式和日志级别
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # 如果不是分布式训练或者是主进程（local_rank为-1或0），则日志级别设置为INFO；否则设置为WARN。
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    # 设置全局随机种子，确保实验的可重复性。
    set_seed(training_args.seed)
    # 设置标签数量为1，这通常用于回归任务或二分类任务。
    num_labels = 1
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False, # 强制使用Python实现的分词器，而非快速（Rust实现）分词器。
    )
    # 加载模型配置
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    # 指定了要使用的模型类
    _model_class = CrossEncoder
    # 加载预训练模型和配置，创建模型实例。
    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    # 训练的数据集
    train_dataset = TrainDatasetForCE(data_args, tokenizer=tokenizer)
    # 指定了训练器类
    _trainer_class = CETrainer
    # 创建trainer实例
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GroupCollator(tokenizer),
        tokenizer=tokenizer
    )
    # 确保输出目录存在
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    # 开始训练过程
    trainer.train()
    # 保存模型
    trainer.save_model()


if __name__ == "__main__":
    main()
