import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

load_dotenv()

def get_file_path(uploaded_file):
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def get_vectorestore_from_url(url):
    file_path = get_file_path(url)
    loader = PyPDFLoader(file_path)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
     # Flatten the list of lists into a single list of strings
    texts = [chunk for sublist in document_chunks for chunk in sublist]
    # create vector stores
    vectore_store =FAISS.from_texts(texts=texts, embedding=OpenAIEmbeddings())
    return vectore_store


def get_context_retriever_chain(vectore_store):
    llm = ChatOpenAI()
    retriever = vectore_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation ,generate search query to look up in order to get information relavent to the conversation",
            ),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question given the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vectore_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke(
        {"chat_history": st.session_state.chat_history, "input": user_query}
    )
    return response["answer"]


st.set_page_config(page_title="chat with Your Pdf Files", page_icon="📚📗")
st.title("Chat With Your Pdf Files 📚📗")
with st.sidebar:
    st.header("settings ✅")
    st.markdown(
        "The application was developed by **Mohammed Bahageel** artificial intelligence scientist as a part of his experiments with retrieval augmentated generation in generative AI, please note that bigger PDF files might result in token errors"
    )
    PDF = st.file_uploader("Upload your pdf file", type=["pdf"])
if PDF is None or PDF == "":
    st.info("**Please Upload your Pdf File 📚📗**")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello how can I help you ?")
        ]
    if "vectore_store" not in st.session_state:
        st.session_state.vectore_store = get_vectorestore_from_url(PDF)
    user_query = st.chat_input("chat with your app")
    if user_query is not None and user_query != "":
        response = get_response(user_query)

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
    # conversation:
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
