import os.path
import torch

# DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- embedding model ---
EMBEDDING_MODEL_PATH = "models/bge-large-zh-v1.5"

# --- embedding API ---
# EMBEDDING_API_URL = "http://127.0.0.1:8080/embed"

# ES召回的结果数量
TOP_N = 1