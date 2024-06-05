import json
from pyexpat import model
import time
import streamlit as st
import os
from glob import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv() 
documents = [] 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pdf_directory = './test_data'

if "model" not in st.session_state: 
    st.session_state["model"] = "gpt-4o" 
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = {}
if "current_history" not in st.session_state:
    st.session_state.current_history = []
if "retriever" not in st.session_state:
    pdf_files = glob(os.path.join(pdf_directory, '*.pdf'))

    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load()
        documents.extend(pdf_documents)

    chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = chunk_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key= OPENAI_API_KEY)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)

    st.session_state.retriever = vectordb.as_retriever()

if "OPENAI_API" not in st.session_state: 
    st.session_state["OPENAI_API"] = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else ""

template = '''
너는 사회복지사의 업무를 도와주기 위한 챗봇이다. 
사회복지 업무와 관련된 메뉴얼과 가이드북을 읽어서 사용자의 질문에 답변할 수 있도록 학습되었다. 
너는 주어진 업무를 아주 잘 한다. 
Answer the question based only on the following context:
{context}
Question: {question}
'''
prompt = ChatPromptTemplate.from_template(template)

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# ChatOpenAI 인스턴스 생성
model = ChatOpenAI(api_key=OPENAI_API_KEY, model=st.session_state["model"], temperature=0)

rag_chain = (
    {'context': st.session_state.retriever | format_docs, 'question': RunnablePassthrough()}  # 'context'는 retriever와 format_docs를 통해 설정되고, 'question'은 그대로 전달됩니다.
    | prompt  # 프롬프트 템플릿을 적용합니다.
    | model  # 모델을 호출합니다. ChatOpenAI 인스턴스에 바인딩된 model 변수를 사용합니다.
    | StrOutputParser()  # 출력 파서를 통해 모델의 출력을 문자열로 변환합니다.
)

if __name__ == '__main__':
    st.title("사회복지 도움 챗봇")
    with st.expander("챗봇 사용방법", expanded=False):
        st.markdown("""
        - 사회복지사의 업무를 도와주는 챗봇입니다.
        - 업무에 필요한 질문을 하고 질문과 답변을 저장할 수 있습니다.
        """)

    with st.sidebar:
        st.title("설정")
        st.session_state["OPENAI_API"] = st.text_input("Enter API Key", value=st.session_state["OPENAI_API"], type="password")

        if st.session_state["OPENAI_API"]:
            st.session_state["model"] = st.selectbox("Select Model", ["gpt-4o", "gpt-3.5-turbo"])

        if st.session_state.chat_history:
            selected_key = st.selectbox("채팅 목록:", options=list(st.session_state.chat_history.keys()))
            if st.button("이전 채팅"):
                uploaded_history = st.session_state.chat_history[selected_key]
                st.session_state.current_history = uploaded_history

    for content in st.session_state.current_history:
        with st.chat_message(content["role"]):
            st.markdown(content['message']) 

    
    
    if chat_input:= st.text_input("Ask a question:"):
        chat_model = ChatOpenAI(api_key=OPENAI_API_KEY, model=st.session_state["model"], temperature=0)

        with st.chat_message("user"):
            st.markdown(chat_input)

        st.session_state.current_history.append({"role": "user", "message": chat_input})
        
        with st.chat_message("ai"):
            response = rag_chain.invoke(chat_input)
            st.write_stream(stream_data(response))

            st.session_state.current_history.append({"role": "ai", "message": response})
            
        if st.button("저장하기"):
            import datetime
            now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            st.session_state.chat_history[now] = st.session_state.current_history

        st.download_button("Download chat as json", data= json.dumps(st.session_state.chat_history, indent=4), mime="text/json")