from embedding.embedding_local import compute_embedding

def test_compute_embedding_with_valid_input():
    text = "这是一个测试文本"
    embedding = compute_embedding(text)
    # 验证返回的嵌入向量是否符合预期
    assert isinstance(embedding, list), "输出必须是一个列表"
    assert len(embedding[0]) == 1024, "向量维度必须是1024维"

# 测试
# pytest test_embedding_local.py -k test_compute_embedding_with_valid_input -v -s