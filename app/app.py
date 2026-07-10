import streamlit as st
from chat.conversation import chat

st.title("Coding Assessment")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "language" not in st.session_state:
    st.session_state.language = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a question about your documents...")
language = st.selectbox(
    "What language will you be coding in today?",
    ("C++", "go", "OpenJDK 21", "OpenJDK 8", "Python 3.14", "Rust", "JavaScript"),
)
if language:
    st.session_state.language = language

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):

        answer = st.write_stream(chat(question, language))
    st.session_state.messages.append({"role": "assistant", "content": answer})
