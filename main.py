from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from collections import Counter

import gradio as gr
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
loader = DirectoryLoader('./gradio/data', glob="*.txt")
documents = loader.load()

chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = chunk_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectordb = Chroma.from_documents(documents=chunks,
                                 embedding=embeddings)
retriever = vectordb.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(api_key=OPENAI_API_KEY,model_name="gpt-4o", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True)

def get_chatbot_response(chatbot_response):
    print(chatbot_response['result'].strip())
    print('\n문서 출처:')
    for source in chatbot_response["source_documents"]:
        print(source.metadata['source'])
        
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="청년 정책 봇")
    msg = gr.Textbox(label="질문해주세요!")
    clear = gr.Button("Clear")
    
    def response(message, chat_history=[]):
        result = qa_chain(message)
        bot_message = result['result']
        bot_message += '\n#source :'
        for i,doc in enumerate(result['source_documents']):
            bot_message += '[' + str(i+1) +']'+ doc.metadata["source"] +'\n'
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(response, inputs=[msg,chatbot], outputs=[msg,chatbot])
    clear.click(lambda: None, None,chatbot,queue=False)
    
demo.launch()