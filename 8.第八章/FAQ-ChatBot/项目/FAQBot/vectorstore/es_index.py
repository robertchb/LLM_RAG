# -*- coding:utf-8 -*-
from elasticsearch import Elasticsearch
from config.es_basic import ES_URL, ES_INDEX_NAME, BATCH_SIZE, DATA_PATH
from elasticsearch import helpers
from embedding.embedding_local import get_embedding
import json
from config.log_config import logger

es = Elasticsearch([ES_URL])

def create_index():
    index_body = {
        "mappings": {
            "properties": {
                "question": {
                    "type": "text"
                },
                "similar_question": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_max_word"
                },
                "similar_question_vector": {
                    "type": "dense_vector",
                    "dims": 1024
                },
                "answer": {
                    "type": "text"
                }
            }
        }
    }

    # 删除已存在的索引（注意：这会删除所有现有数据！）
    if es.indices.exists(index=ES_INDEX_NAME):
        response = es.indices.delete(index=ES_INDEX_NAME)
        logger.info(f"------ 删除索引响应：{response} ------")
        if response.get("acknowledged") != True:
            logger.error("------ 索引删除未被确认，请检查问题原因 ------")
            return

    # 创建新的索引
    response = es.indices.create(index=ES_INDEX_NAME, body=index_body)
    if response.get("acknowledged") == True:
        logger.info("------ 医疗索引创建完毕 ------")
    else:
        logger.error("------ 索引创建未被确认，请检查问题原因 ------")


# 批量将数据写入索引
def read_jsonl_file(file_path: str):
    """一批一批地读取数据"""
    batch = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            # 跳过空行
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
                # 循环遍历每个similar_question
                for sim_q in item["similar_question"]:
                    # 计算句向量
                    sim_q_vec = get_embedding(sim_q) # shape : 1024
                    # 构造数据项
                    record = {
                        "question": item["question"],
                        "similar_question": sim_q,
                        "similar_question_vector": sim_q_vec,
                        "answer": item["answer"]
                    }
                    # 构造action
                    action = {
                        "_index": ES_INDEX_NAME,
                        "_source": record
                    }
                    # 暂存
                    batch.append(action)

                    # 当达到batch_size大小时，返回当前批次的数据
                    if len(batch) == BATCH_SIZE:
                        yield batch
                        batch = []
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误： {e}")
            except Exception as e:
                print(f"未知报错：{e}")

        # 返回文件末尾的最后一批数据（如果有的话）
        if batch:
            yield batch


# 使用helpers.bulk方法批量插入数据
def bulk_insert(file_path: str):
    try:
        for batch in read_jsonl_file(file_path):
            response = helpers.bulk(es, batch)
            print("### 批量插入成功", response)
    except helpers.BulkIndexError as e:
        print("### 批量插入异常：{e}", e.errors)

def main():
    # 第1步：创建索引
    create_index()
    # 第2步：向索引写入数据
    bulk_insert(DATA_PATH)


if __name__ == "__main__":
    main()

# 测试
# python vectorstore/es_index.py