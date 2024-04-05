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
import base64
from openai import OpenAI

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
    # create vector stores
    vectore_store =FAISS.from_documents(document_chunks, OpenAIEmbeddings())
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
    response = conversation_rag_chain.stream(
        {"chat_history": st.session_state.chat_history, "input": user_query}
    )
    for chunk in response_stream:
        content=chunk.get("answer","")
        yield content
# convert text back to audio
def text_to_audio(client, text, audio_path):
    response = client.audio.speech.create(model="tts-1", voice="fable", input=text)
    response.stream_to_file(audio_path)
client = OpenAI()
# autoplay audio function
def autoplay_audio(audio_file):
    with open(audio_file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    audio_html = (
        f'<audio src="data:audio/mp3;base64 ,{base64_audio}" controls autoplay>'
    )
    st.markdown(audio_html, unsafe_allow_html=True)


st.set_page_config(page_title="chat with Your Pdf Files", page_icon="📚📗")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    title="Retrieval Augmented Generation"
    name = " Developed by:Mohammed Bahageel"
    profession = "Artificial Intelligence developer"
    imgUrl="https://image.similarpng.com/very-thumbnail/2020/07/Pharmacy-logo-vector-PNG.png"
    styles={
            "container": {"padding": "0!important", "background-color": "white" },
            "icon": {"color": "red", "font-size": "18px" }, 
            "nav-link": {"font-size": "18px", "text-align": "center", "margin":"0px", "--hover-color": "hsl(264, 100%, 61%)"},
            "nav-link-selected": {"background-color": " hsl(264, 100%, 61%)"},
        }
    
    st.markdown(
        f"""
        <div class="st-emotion-cache-18ni7ap ezrtsby2">
            <a href="{imgUrl}">
                <img class="profileImage" src="{imgUrl}" alt="Your Photo">
            </a>
            <div class="textContainer">
                <div class="title"><p>{title}</p></div>
                <p>{name}</p>
                <p>{profession}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(
                content=" Hello ! with you is RAG chatbot  how can I assist you today ? 🥰"
            )
        ]
   
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
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human", avatar="👨‍⚕️"):
            st.markdown(user_query)
        with st.chat_message("AI", avatar="🤖"):
            response=st.write_stream(get_response(user_query))
            response_audio_file = "audio_response.mp3"
            text_to_audio(client, response, response_audio_file)
            autoplay_audio(response_audio_file)
            st.session_state.chat_history.append(AIMessage(content=response))
       
    # conversation:
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
