import requests
from requests.exceptions import RequestException
import json
import random

def test_faq_search(port):
    # 服务的URL
    url = f"http://127.0.0.1:{port}/faq/search"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    # 模拟请求数据
    query_data = {"text": "我最近长胖了，这让我有些尴尬，是否可以改善这一状况？"}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(query_data))
        # 检查响应状态码
        assert response.status_code == 200, "响应报错，服务异常。"
        print(f"Port:{port}, response: {response.json()}")
    except RequestException as e:
        print("端口：{port},  {e} 上的服务请求失败。")

def main():
    # 端口列表，对应三个服务
    ports = [7001, 7002, 7003]
    # 测试100次
    num_tests = 100
    for _ in range(num_tests):
        port = random.choice(ports)
        test_faq_search(port)

if __name__ == "__main__":
    main()



# 测试
#  python test_server_docker-compose.py