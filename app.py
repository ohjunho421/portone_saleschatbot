import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import glob

# 1. 환경 변수 로드
load_dotenv()

# 웹사이트에서 텍스트 추출
def get_website_text(urls):
    documents = []
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = '\n'.join(chunk for chunk in chunks if chunk)
            # 각 URL별로 Document 객체를 생성하여 출처(source)를 메타데이터에 저장
            documents.append(Document(page_content=clean_text, metadata={"source": url}))
        except Exception as e:
            st.warning(f"'{url}' URL 처리 중 오류 발생: {e}")
    return documents

# PDF 문서에서 텍스트 추출
def get_pdf_text(pdf_docs):
    documents = []
    if not pdf_docs:
        return []
    for pdf_doc in pdf_docs:
        try:
            with open(pdf_doc.name, "wb") as f:
                f.write(pdf_doc.getbuffer())
            
            loader = PyMuPDFLoader(file_path=pdf_doc.name)
            # load()는 Document 객체의 리스트를 반환합니다.
            documents.extend(loader.load())
            
            os.remove(pdf_doc.name)
        except Exception as e:
            st.warning(f"'{pdf_doc.name}' 파일 처리 중 오류 발생: {e}")
    return documents

# 텍스트를 청크로 분할
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# 텍스트 청크로 벡터 저장소 생성
def get_vector_store(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # from_texts 대신 from_documents를 사용하여 메타데이터를 함께 저장
    vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vector_store

# 대화 체인 생성 (최신 LangChain 방식)
def get_conversation_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 포트원(PortOne) 서비스 전문 안내원입니다. 제공된 내용을 바탕으로 사용자의 질문에 친절하고 명확하게 답변해주세요.

주어진 내용(Context)에서 답변을 찾을 수 없는 경우, "죄송합니다. 문의하신 내용은 제가 답변해드리기 어려운 부분입니다."라고 답변한 후, 다음과 같이 미팅을 제안해주세요.
"더 자세한 내용이 궁금하시거나 기술 도입에 대한 논의가 필요하시면, ocean@portone.io 로 메일 주시면 미팅을 통해 상세히 안내해 드리겠습니다."

내용을 찾을 수 있는 경우에는, 미팅 제안 없이 질문에 대한 답변만 제공해주세요.

Context:
{context}
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vector_store.as_retriever(), question_answer_chain)
    
    return rag_chain

# 데이터 처리 및 대화 체인 생성 (캐싱 적용)
def process_data_and_get_chain():
    try:
        # Add your URLs here
        urls = [
            'https://portone.io/korea/ko?gad_source=1&gad_campaignid=21635208142&gbraid=0AAAAACP78VtcsdeBYpVbJTPkM4c-YV1ZR&gclid=CjwKCAjwprjDBhBTEiwA1m1d0kBeEwEJ1WRgaPc0BHTk6bmjz4YoOKg5Dx2uonZlh3YUWwc_01wo2hoCK6wQAvD_BwE',
            'https://help.portone.io/?_gl=1*6zmbto*_gcl_aw*R0NMLjE3NTIxMTYwMDQuQ2p3S0NBandwcmpEQmhCVEVpd0ExbTFkMGtCZUV3RUoxV1JnYVBjMEJIVGs2Ym1qejRZb09LZzVEeDJ1b25abGgzWVVXd2NfMDF3bzJob0NLNndRQXZEX0J3RQ..*_gcl_au*ODkxMzU4MjcuMTc0OTQzMjg4Ni4xNDQ2ODk5NTg4LjE3NTAwNTgzNDUuMTc1MDA1ODM0NQ..*_ga*MTM2NjQzMDUwNy4xNzQ5NDMyODkw*_ga_PD0FDL16NZ*czE3NTIxMTYwMDMkbzE0JGcxJHQxNzUyMTE2MDA5JGo1NCRsMCRoMA..',
            'https://developers.portone.io/opi/ko/readme?v=v2',
            'https://blog.portone.io/?_gl=1*19o3f43*_gcl_aw*R0NMLjE3NTIxMTYwMDQuQ2p3S0NBandwcmpEQmhCVEVpd0ExbTFkMGtCZUV3RUoxV1JnYVBjMEJIVGs2Ym1qejRZb09LZzVEeDJ1b25abGgzWVVXd2NfMDF3bzJob0NLNndRQXZEX0J3RQ..*_gcl_au*ODkxMzU4MjcuMTc0OTQzMjg4Ni4xNDQ2ODk5NTg4LjE3NTAwNTgzNDUuMTc1MDA1ODM0NQ..*_ga*MTM2NjQzMDUwNy4xNzQ5NDMyODkw*_ga_PD0FDL16NZ*czE3NTIxMTYwMDMkbzE0JGcwJHQxNzUyMTE2MDAzJGo2MCRsMCRoMA..'
        ]
        
        pdf_docs = get_pdf_text(st.session_state.get("pdf_docs", []))
        website_docs = get_website_text(urls)
        
        all_docs = pdf_docs + website_docs
        if not all_docs:
            return None

        # 텍스트 분할기가 Document 객체를 처리하도록 수정
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        
        vector_store = get_vector_store(splits)
        conversation_chain = get_conversation_chain(vector_store)
        return conversation_chain
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
        st.warning("Google API 키가 .env 파일에 올바르게 설정되었는지 확인해주세요.")
        return None

# 메인 함수
def main():
    st.set_page_config(page_title="포트원 챗봇 (Gemini)", page_icon=":robot_face:")
    st.header("포트원(PortOne) 서비스 안내 챗봇")
    st.write("안녕하세요! 포트원 서비스에 대해 궁금한 점을 무엇이든 물어보세요.")

    conversation_chain = process_data_and_get_chain()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
                if message.metadata.get("sources"):
                    st.markdown("---_답변 출처_---")
                    for source in message.metadata["sources"]:
                        st.markdown(source)

    if prompt := st.chat_input("포트원 서비스에 대해 질문해주세요."):
        with st.chat_message("user"):
            st.markdown(prompt)

        if conversation_chain is None:
            st.warning("데이터 로딩에 실패했습니다. PDF 파일이 있는지, API 키가 올바른지 확인 후 페이지를 새로고침해주세요.")
            return

        with st.spinner("답변을 생성 중입니다..."):
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            response = conversation_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history
            })
            answer = response['answer']
            sources = set(doc.metadata.get("source") for doc in response["context"] if doc.metadata.get("source"))
            st.session_state.chat_history.append(AIMessage(content=answer, metadata={"sources": list(sources)}))
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources:
                    st.markdown("---_답변 출처_---")
                    for source in sources:
                        st.markdown(source)

if __name__ == '__main__':
    main()