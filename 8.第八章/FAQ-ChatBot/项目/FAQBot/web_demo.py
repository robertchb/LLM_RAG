import streamlit as st
import time
import requests

# 设置页面配置
st.set_page_config(page_title="简答FAQ问答系统")
st.title("简单FAQ问答系统")

# API的URL
API_URL = "http://127.0.0.1:8008/faq/search"

def response_generator(data):
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        answer = response.json().get("answer", "抱歉，找不到答案。")
    else:
        answer = "抱歉，无法获取答案。"
    for word in answer:
        yield word
        time.sleep(0.05)

if "history" not in st.session_state:
    st.session_state.history = []

if st.session_state.history:
    for pair in st.session_state.history:
        with st.chat_message(pair["role"]):
            st.markdown(pair["content"])

if prompt := st.chat_input("请输入..."):
    st.session_state.history.append({"role": "user",
                                     "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 用户输入
        data = {"text": prompt}
        response = st.write_stream(response_generator(data))

    st.session_state.history.append({"role": "assistant",
                                     "content": response})


# 运行命令：
# streamlit run web_demo.py --server.port 8502

# 测试问题：我的肚子比较大，如何减肥呀？

# 测试问题：这家伙晚上睡觉经常磨牙，怎么办呀？