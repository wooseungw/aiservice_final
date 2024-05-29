# 필요한 라이브러리 및 모듈을 임포트합니다.
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# 문서들을 형식화하는 함수를 정의합니다.
# 각 문서의 페이지 내용을 합쳐 하나의 문자열로 반환합니다.
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

### gpt를 기반으로 한 챗봇을 만들기 위한 클래스를 정의합니다. 이 클래스의 출력은 string형태 입니다.
class chaingpt:
    def __init__(self,api_key,retriever, sys_prompt="",model="gpt-4o"):
        self.template = sys_prompt + '''Answer the question based only on the following context:
        {context}

        Question: {question}
        '''
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.model = ChatOpenAI(api_key=api_key,model=model, temperature=1)
        self.chainmodel = (
        {'context': retriever |self._format_docs, 'question': RunnablePassthrough()}  # 'context'는 retriever와 format_docs를 통해 설정되고, 'question'은 그대로 전달됩니다.
        | self.prompt  # 프롬프트 템플릿을 적용합니다.
        | self.model  # 모델을 호출합니다.
        | StrOutputParser()  # 출력 파서를 통해 모델의 출력을 문자열로 변환합니다.
        )
        
    ### 내부에서 사용하는 함수는 _로 시작합니다.
    def _format_docs(self,docs):
        return '\n\n'.join(doc.page_content for doc in docs)
    
    ### 모델의 출력을 반환하는 함수를 정의합니다.
    def invoke(self,input_message):
        return self.chainmodel.invoke(input_message)
    def stream(self,input_message):
        return self.chainmodel.stream(input_message)