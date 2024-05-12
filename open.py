import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
import openai

# Set your OpenAI API key
openai.api_key = "API ky Open AI"
def split_text_into_chunks(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def handle_user_input(user_question, memory):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=user_question,
        max_tokens=150
    )
    st.session_state.chat_history.append(response.choices[0].text.strip())
    memory.extend(st.session_state.chat_history)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your GPT")
    st.write(css, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    memory = st.session_state.chat_history

    st.header("Nova Bot")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question, memory)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = split_text_into_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

if __name__ == '__main__':
    main()
