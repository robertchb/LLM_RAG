import requests

# 要测试的端点URL
url = "http://localhost:8000/embeddings/"

# 准备请求的数据
data_1 = {
    "text": "今天天气很好"
}

data_2 = {
    "text": ["生成式大语言模型", "自然语言处理"]
}


# 发送POST请求
response = requests.post(url, json=data_1)
#response = requests.post(url, json=data_2)

# 检查响应
if response.status_code == 200:
    # 正常响应
    result = response.json()
    print("返回的结果数量: ", len(result.get("embeddings")))
    print("嵌入向量: ", result.get("embeddings")[0][:5])
else:
    print(f"请求失败，状态码: {response.status_code}, 响应内容: {response.text}")