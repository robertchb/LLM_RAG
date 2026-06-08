# 从gevent库导入monkey模块并调用patch_all()方法。
# 这个方法会修改标准库中的很多阻塞调用，使之变成非阻塞的，从而利用gevent的协程实现并发。
# 这对于提高IO密集型应用的性能很有帮助。
from gevent import monkey
monkey.patch_all()

# 对于生产环境，保持为False
debug = False

# 指定应用绑定的IP地址和端口号。0.0.0.0表示监听所有的网络接口，7000是指定的端口号。
bind = "0.0.0.0:7000"

# 监听队列，设置操作系统可以挂起的最大连接数量为2048。这是等待被服务器接受的连接的数量上限。
backlog=10 # 2048

# gevent是基于协程的，有时候设置一些线程（比如2-4个）可以帮助处理某些阻塞操作，
# 从而不会阻塞整个进程。然而，对于大多数基于gevent的应用，保持默认的1个线程就足够了，
# 因为gevent的协程已经提供了很好的并发性。
threads=1

# 每个worker可以同时处理的最大连接数
worker_connections=10 # 1000

# 对于FAQ chatbot这样的I/O密集型应用，进程数可以设置为CPU核心数的2-4倍。
workers = 1 # 2

# gevent使用协程，允许单个Python进程处理数千个并发连接，非常适合网络密集型应用。
worker_class = "gevent"

# 设置超时时间为300秒。如果一个工作进程在这段时间内没有响应，它将会被重启。
timeout=300
