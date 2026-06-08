# 使用Python镜像作为基础
FROM python:3.11.0-slim
# 设置时区环境变量，避免运行时配置
ENV TZ=Asia/Shanghai
# 利用Docker的多阶段构建特性，减小最终镜像大小
# 首先是构建阶段
FROM python:3.11.0-slim as builder
WORKDIR /build
COPY requirements.txt .
# 使用pip wheel预先构建依赖，这样做可以减少最终镜像中不必要的文件
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt


# 第二阶段，然后是最终镜像
FROM python:3.11.0-slim
# 设置工作目录
WORKDIR /code
# 将时区设置为上海，利用环境变量进行配置
ENV TZ=Asia/Shanghai

# 将从builder阶段构建的wheels和代码复制到最终镜像中
COPY --from=builder /build/wheels /wheels
COPY . /code
# 安装依赖（仅安装预构建的wheels，减少了构建时间和最终镜像大小）
RUN pip install --no-cache-dir /wheels/*

# 容器在运行时会监听端口
EXPOSE 7000

# 设置默认命令，Docker容器启动时执行以下操作：

# 方式-1：
# 使用gunicorn作为Web服务器来运行Python web应用
# -c：这是gunicorn的一个命令行选项，指定配置文件的路径
# /config/server_config.py：这是gunicorn的配置文件的路径
# server:app：这指定了gunicorn应该运行的应用对象
# CMD ["gunicorn", "-c", "config/server_config.py", "server:app"]

# 方式-2：
# uvicorn 是一个轻量级、超快的异步服务器网关接口 (ASGI) 服务器，适用于异步应用。
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7000"]