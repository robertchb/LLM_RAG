import requests
import time
# ThreadPoolExecutor是一个线程池执行器，用于并发执行提供的调用。
# as_completed是一个函数，用于返回一个迭代器。
from concurrent.futures import ThreadPoolExecutor, as_completed

# 测试服务端点
URL = "http://localhost:8000/embeddings/"

# 请求的数据
DATA = {
    "text": "今天天气非常好!"
}

# 并发请求数量，总的请求数
REQUESTS = 100

# 同时并发的请求数
CONCURRENT_REQUESTS = 10

def send_request(url, data):
    '''发送请求并返回响应时间'''
    start_time = time.time()
    # json=data，将data作为JSON格式的数据体发送
    response = requests.post(url, json=data)
    cost_time = time.time() - start_time
    
    return cost_time, response.status_code

def main(url, data, total_request, concurrent_requests):
    start_time = time.time()
    # 创建一个ThreadPoolExecutor实例
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        # 使用列表推导式，为每个请求创建一个future对象并存储在列表futures中。
        # executor.submit方法用于提交执行函数
        futures = [executor.submit(send_request, url, data) for _ in range(total_request)]
        succeeded = 0
        total_cost = 0
        # 使用as_completed(futures)迭代完成的future对象
        for future in as_completed(futures):
            # 获取每个请求的响应时间和状态码
            cost_time, status = future.result()
            # 统计成功的请求次数
            if status == 200:
                succeeded += 1
            total_cost += cost_time
    end_time = time.time()
    # 总的测试时间
    total_time = end_time - start_time
    print(f"成功请求次数：{succeeded} / {total_request}")
    print(f"总测试时间：{total_time:.2f}秒")
    
    if succeeded > 0:
        avg_response_time = total_cost / succeeded
        print(f"平均响应时间：{avg_response_time:.2f}秒")
        qps = succeeded / total_time
        print(f"QPS(每秒查询率): {qps:.2f}")
        
if __name__ == "__main__":
    main(URL, DATA, REQUESTS, CONCURRENT_REQUESTS)