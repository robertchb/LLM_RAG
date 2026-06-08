import requests
import json

def test_faq_search():
    # 服务的URL
    url = "http://127.0.0.1:8008/faq/search"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    # 模拟请求数据
    query_data = {"text": "我最近长胖了，这让我有些尴尬，是否可以改善这一状况？"}
    response = requests.post(url, headers=headers, data=json.dumps(query_data))
    # 检查响应状态码
    assert response.status_code == 200, "响应报错，服务异常。"
    print(response.json())

# 测试
#  pytest test_server.py -k test_faq_search -v -s