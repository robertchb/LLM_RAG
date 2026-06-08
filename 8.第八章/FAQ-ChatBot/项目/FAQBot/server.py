#-*- coding: utf-8 -*-
# @Date: 2024-03-20 20:40:00
# @Author: 唐国梁Tommy

from embedding.embedding_local import get_embedding
from config.basic_config import TOP_N
from vectorstore.es_match import ESRecall

import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.log_config import logger

# 创建应用实例
app = FastAPI()

class QueryData(BaseModel):
    '''数据类模型'''
    text: str

@app.post("/faq/search")
def search(query_data: QueryData):
    try:
        logger.info(f"用户输入的文本: {query_data.text}")
        # 计算query的向量表示
        query_vec = get_embedding(query_data.text)
        # 基于BM25和密集检索进行召回
        results = ESRecall.recall_by_bm25_dr(query_data.text,
                                             query_vec,
                                             topN=TOP_N)
        if results:
            answer = results[0].get("answer")
            logger.info(f"匹配的回答：{answer}")
            return {"answer" : answer}
        else:
            raise HTTPException(status_code=404, detail="没有找到匹配的结果")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)

# 在命令行窗口执行命令：
# python server.py

# 服务测试命令：
# curl -X 'POST' 'http://127.0.0.1:7000/faq/search' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"text": "我最近长胖了，这让我有些尴尬，是否可以改善这一状况？"}'