# loguru是一个第三方库，提供了一个简单但强大的日志管理系统。
from loguru import logger
import os
import sys
from pathlib import Path

# 确保日志文件存储的目录存在
log_directory = (Path(__file__).parent / "../logs").resolve()
# 检查日志目录是否存在
log_directory.mkdir(parents=True, exist_ok=True)
# 通过环境变量LOG_LEVEL获取日志级别，默认为INFO
log_level = os.getenv("LOG_LEVEL", "INFO")
# 日志的格式，这个格式字符串指定了日志的时间、级别、来源（包括模块名、函数名和行号）以及日志消息的显示格式。
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level>  | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# 配置了一个日志处理器，用于将日志输出到一个文件中。
logger.add(
    log_directory / "faq_chatbot_{time:YYYY-MM-DD_HH-MM}.log",
    rotation="2 hours", # 每2小时轮转一次日志文件
    retention="10 days", # 保留最近10天的日志文件
    compression="zip", # compression="zip"表示压缩旧的日志文件为zip格式
    level=log_level, # 日志级别
    format=log_format, # 日志格式
    enqueue=True # 启用异步记录
)

# 配置了一个日志处理器，用于将日志输出到标准错误流sys.stderr
logger.add(
    sys.stderr,
    format=log_format,
    level=log_level
)

# def catch_unhandled_exceptions(ex_type, ex_value, ex_traceback):
#     '''用于捕获未处理的异常'''
#     # 如果是KeyboardInterrupt（即用户中断执行），则调用系统的默认异常处理钩子。
#     if issubclass(ex_type, KeyboardInterrupt):
#         sys.__excepthook__(ex_type, ex_value, ex_traceback)
#         return
#
#     logger.exception("Unhandled exception: ", exc_info=(ex_type,
#                                                         ex_value,
#                                                         ex_traceback))
#
# # 将sys.excepthook设置为上面定义的函数。
# # sys.excepthook是一个全局变量，当程序出现未捕获的异常时，Python会调用这个钩子。
# sys.excepthook = catch_unhandled_exceptions
