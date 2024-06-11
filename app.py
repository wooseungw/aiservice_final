import streamlit as st
import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

import time
import datetime
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

documents = []

def update_prompt(sub):
    if sub == "manual":
        prompt = """
        너는 매뉴얼을 읽고 사용자가 필요한 정보를 제공기능에 특화되어있어. 업무 매뉴얼은 이전년도와 현재 변경 내용을 담고있어 그러니 너는 뭐가 변경되었는지 알려줘야해.
        예를들어
        "작년과 비교해서 올해 변경된 내용을 알려줘" 와 같은 내용을 물어본다면,
        
        24년도의 변경점은 다음과 같습니다.
        전년도 OOOOOOOO 했던 내용은 OOOO로 변경되었으며, 상세한 내용은 ... 입니다.
        """
    elif sub == "bi":
        prompt = """
        너는 저장되어있는 사업안내를 읽고 사용자가 필요한 정보를 제공기능에 특화되어있어. 
        
        어떤 사업을 시행했고, 서비스 대상이 누구인지, 지원금액은 얼마인지 등의 정보를 굉장히 구체적으로 제공할 수 있어야해.
        """
    return prompt

def update_retriever(sub):
        vectordb = st.session_state.dbs[sub]
        return vectordb.as_retriever()
        

if "OPENAI_API" not in st.session_state:
    st.session_state["OPENAI_API"] = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else ""
# 기본 모델을 설정합니다.
if "model" not in st.session_state:
    st.session_state["model"] = "gpt-4o"
# 채팅 기록을 초기화합니다.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "previous_history" not in st.session_state:
    st.session_state.previous_history = " "

if "previous_bool"  not in st.session_state:
    st.session_state.previous_bool = False

if "sub" not in st.session_state:
    st.session_state['sub'] = "manual"
    
if "sys_prompt" not in st.session_state:
    st.session_state.sys_prompt = ''' '''

if "dbs" not in st.session_state:
    st.session_state.dbs = {}
    pdf_directorys = ['./manual', './bi']
    embeddings = OpenAIEmbeddings(api_key=st.session_state["OPENAI_API"])
    
    for pdf_directory in pdf_directorys:
        pdf_files = glob(os.path.join(pdf_directory, '*.pdf'))
        # Load all PDF files using PyPDFLoader
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            pdf_documents = loader.load()
            documents.extend(pdf_documents)
        # 텍스트는 RecursiveCharacterTextSplitter를 사용하여 분할
        chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunk = chunk_splitter.split_documents(documents)
        vectordb = Chroma.from_documents(documents=chunk, embedding=embeddings)
        st.session_state.dbs[pdf_directory.split('/')[-1]] = vectordb
        print(pdf_directory," Chunks split Done.")
        print("Retriever Done.")
if "retriever" not in st.session_state:
    st.session_state.retriever = None
    
################# 설정을 위한 사이드바를 생성합니다. 여기서 api키를 받아야 실행됩니다. ##########################################
with st.sidebar:
    st.title("설정")
    st.session_state["OPENAI_API"] = st.text_input("Enter API Key", st.session_state["OPENAI_API"], type="password")
    st.session_state["model"] = st.selectbox("Select Model", ["gpt-4o", "gpt-3.5-turbo"])

    subject = st.radio("주제 선택", ["매뉴얼+가이드북", "사업안내"],horizontal=True)
    if subject == "매뉴얼+가이드북":
        st.session_state["sub"] = "manual"
    elif subject == "사업안내":
        st.session_state["sub"] = "bi"
    # 주제에 따라 프롬프트를 업데이트합니다.
    st.session_state.sys_prompt = update_prompt(st.session_state["sub"])
    # 주제에 따라 retriever를 업데이트합니다.
    st.session_state.retriever =  update_retriever(st.session_state["sub"])
    
    st.session_state.previous_history = st.selectbox("이전 대화내용 불러오기", options=[" "] + list(os.listdir("previous_chat") if os.path.exists("previous_chat") else None))
    if st.session_state.previous_bool == False:
        if st.session_state.previous_history != " ":
            with open(f"previous_chat/{st.session_state.previous_history}", "r") as f:
                st.session_state.previous_history = json.load(f)
            st.session_state.chat_history = st.session_state.previous_history + st.session_state.chat_history 
            st.session_state.previous_bool = True

# 주제에 따라 챗봇의 제목과 사용법을 업데이트합니다.
if st.session_state['sub'] == "manual":
    st.title("사회복지 매뉴얼 챗봇")
    with st.expander("챗봇 사용법", expanded=False):
        st.markdown("""
                    - 이 챗봇은 방대한 양의 매뉴얼을 모두 숙지하기 어려운 사회복지사의 업무를 도와주기 위한 챗봇입니다.
                    - 답변의 내용은 사회복지사의 업무와 관련된 매뉴얼과 가이드북을 학습하여 생성됩니다.
                """)
elif st.session_state['sub'] == "bi":
    st.title("사회복지 사업안내 챗봇")
    with st.expander("챗봇 사용법", expanded=False):
        st.markdown("""
                    - 이 챗봇은 여러 사업안내를 모두 숙지하기 어려운 사회복지사의 업무를 도와주기 위한 챗봇입니다.
                    - 답변의 내용은 데이터베이스에 등록된 사업안내를 학습하여 생성됩니다.
                    """)

# 프롬프트 템플릿을 정의합니다.
# SYS_PROMPT는 시스템 메시지로, 템플릿에 포함됩니다. 
# {context}와 {question}은 실행 시 동적으로 채워질 자리표시자입니다.
template = f'''
너는 사회복지사의 업무를 도와주기 위한 챗봇이야. 성실하게 답변하면 10달러를 줄게. \\
사회복지 업무와 관련된 매뉴얼과 가이드북을 바탕으로 사용자의 질문에 답변할 수 있도록 학습되어있어. \\
뭐든 천천히 생각하고 답변해줘야해. \\

{st.session_state.sys_prompt}
'''+'''
Answer the question based only on the following context:
{context}

Question: {question}
'''
# ChatPromptTemplate.from_template() 메서드를 사용하여 프롬프트 템플릿을 생성합니다.
prompt = ChatPromptTemplate.from_template(template)
################## 챗봇을 사용하기 위한 gpt 모델을 정의합니다. ############################################################
# ChatOpenAI 인스턴스를 생성하여 LLM (대규모 언어 모델)을 설정합니다.
# 여기서는 'gpt-4o' 모델을 사용하고, temperature는 0으로 설정하여 출력의 일관성을 높입니다.
model = ChatOpenAI(api_key=st.session_state["OPENAI_API"],
                    model=st.session_state["model"], 
                    temperature=0
                    )
# 문서들을 형식화하는 함수를 정의합니다.
# 각 문서의 페이지 내용을 합쳐 하나의 문자열로 반환합니다.
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# RAG (Retrieval-Augmented Generation) 체인을 연결합니다.
# 이 체인은 문서 검색, 형식화, 프롬프트 적용, 모델 호출, 출력 파싱의 과정을 거칩니다.
rag_chain = (
    {'context': st.session_state.retriever | format_docs, 'question': RunnablePassthrough()}  # 'context'는 retriever와 format_docs를 통해 설정되고, 'question'은 그대로 전달됩니다.
    | prompt  # 프롬프트 템플릿을 적용합니다.
    | model  # 모델을 호출합니다.
    | StrOutputParser()  # 출력 파서를 통해 모델의 출력을 문자열로 변환합니다.
)

############################################ 실제 챗봇을 사용하기 위한 Streamlit 코드 ###################################################
for content in st.session_state.chat_history:
    with st.chat_message(content["role"]):
        st.markdown(content['message'])
        
### 사용자의 입력을 출력하고 생성된 답변을 출력합니다.
if prompt := st.chat_input("질문을 입력하세요."):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("ai"):
            if len(st.session_state.chat_history) > 2:
                querry = "이건 이전대화야"+str(st.session_state.chat_history[-2:]) + f"\n\n 이건 내 질문이야.{prompt}"
                response = rag_chain.invoke(querry)
            else:
                querry = f"\n\n 이건 내 질문이야.{prompt}"
                response = rag_chain.invoke(querry)
            
            st.write_stream(stream_data(response))
        
        st.session_state.chat_history.append({"role": "user", "message": prompt})
        st.session_state.chat_history.append({"role": "ai", "message": response})
 

with st.sidebar:   
    c1,c2 = st.columns(2)
    with c1:
        if st.button("대화내용 저장"):
            st.session_state.previous_history = st.session_state.chat_history
            with open(f"previous_chat/{now}.json", "w") as f:
                json.dump(st.session_state.chat_history, f, indent=4)
    with c2:
        st.download_button("json 다운로드", json.dumps(st.session_state.chat_history, indent=4).encode('utf-8'), f"chat_history_{now}.json", "json")
    

