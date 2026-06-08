
from transformers import AutoTokenizer, AutoModel
from config.basic_config import DEVICE, EMBEDDING_MODEL_PATH
import torch
from config.log_config import logger

# 加载预训练模型
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH).to(DEVICE)
model.eval()

def compute_embedding(text: str):
    '''异步计算文本列表的句子嵌入向量'''
    try:
        # 将文本编码为模型可以处理的格式
        encoded_input = tokenizer(text,
                                  padding=True,
                                  truncation=True,
                                  return_tensors="pt",
                                  max_length=512).to(DEVICE)
        with torch.no_grad():
            # 模型推理
            outputs = model(**encoded_input)
            # 从模型输出中提取句子嵌入，获取输出的最后一层的隐藏状态并选择每个序列的第一个token（通常是[CLS]）的嵌入
            sentence_embedding = outputs[0][:, 0]
            # L2标准化句子嵌入
            sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)

        # 将句子嵌入从PyTorch张量转换为numpy数组，然后转换为列表，作为函数的返回值。
        return sentence_embedding.cpu().tolist()
    except Exception as e:
        # exc_info=True : 告诉日志记录器除了记录错误消息外，还要将异常的信息包括在日志消息中。
        logger.error(f"Errors in compute embedding : {e}", exc_info=True)

def get_embedding(text: str):
    embedding = compute_embedding(text)
    return embedding[0]