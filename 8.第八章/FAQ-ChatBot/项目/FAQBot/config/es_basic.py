
# Elasticsearch URL
# 1. 在往索引中写入数据（知识库）时，用这个地址
#ES_URL = "http://0.0.0.0:9200"

# 2. 数据插入之后，启动应用服务时，用这个地址。记得要切换一下 ^_^
ES_URL = "http://172.18.0.2:9200"

# ES INDEX
ES_INDEX_NAME = "db"

# 数据批次大小
BATCH_SIZE = 10

# 数据集路径
DATA_PATH = "data/samples.jsonl"