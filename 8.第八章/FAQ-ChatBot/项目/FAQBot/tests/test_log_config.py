from config.log_config import logger
import os

def get_current_log_file_path():
    log_directory = "/logs"
    log_files = os.listdir(log_directory)
    latest_log_file = max([os.path.join(log_directory, f) for f in log_files], key=os.path.getctime)
    return latest_log_file

def test_log_message():
    log_message = "测试日志记录功能"
    logger.info(log_message)
    log_file_path =get_current_log_file_path()
    with open(log_file_path, "r") as log_file:
        logs = log_file.read()
        assert log_message in logs

# 测试命令：
#