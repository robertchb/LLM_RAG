# -*- coding: utf-8 -*-

from elasticsearch import Elasticsearch
from config.es_basic import ES_URL, ES_INDEX_NAME
from config.basic_config import TOP_N
from vectorstore.es_index import get_embedding
from config.log_config import logger

es = Elasticsearch([ES_URL])

class ESRecall:
    @staticmethod
    def recall_by_bm25_dr(text: str, text_vec: list,  topN: int = 3):
        """
        使用BM25和密集检索进行召回
        :param text: 查询文本
        :param text_vec: 查询文本的向量表示
        :param topN: 返回的文档数目
        :return: ES的查询结果
        """
        if not text or not text_vec:
            raise ValueError("文本和文本向量不能为空")

        condition = {
            "query": {
                "function_score": {
                    "query": {
                        "match": {"similar_question": text}
                    },
                    "functions": [
                        {
                            "script_score": {
                                "script": {
                                    "source": "cosineSimilarity(params.vec, 'similar_question_vector') + 1.0",
                                    "params": {"vec": text_vec}
                                }
                            }
                        }
                    ]
                }
            }
        }

        try:
            result = es.search(index=ES_INDEX_NAME,
                               body=condition,
                               size=topN,
                               _source=["question", "similar_question", "answer"]
                               )
            final_result = [item["_source"] for item in result["hits"]["hits"]]
            return final_result
        except Exception as e:
            logger.error(f"搜索失败，抛出异常：{e}")
            return []


if __name__ == "__main__":
    # 测试样本
    text = "我最近长胖了，这让我有些尴尬，是否可以改善这一状况？"
    # 计算样本向量
    text_vec = get_embedding(text)
    # 基于BM25和向量进行召回
    res = ESRecall.recall_by_bm25_dr(text, text_vec, topN=TOP_N)
    print(res)
