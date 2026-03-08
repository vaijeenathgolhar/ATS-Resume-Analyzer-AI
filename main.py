# ================== IMPORTS ==================
import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

# ================== LOAD ENV ==================

load_dotenv()

# ================== LLM ==================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

parser = StrOutputParser()

# ================== STREAMLIT ==================
st.set_page_config(page_title="ATS Resume Scanner", layout="wide")
st.title("🚀 ATS Resume Analyzer with Memory")

st.sidebar.header("Upload Files")

resume_file = st.sidebar.file_uploader("Upload Resume", type="pdf")
jd_file = st.sidebar.file_uploader("Upload Job Description", type="pdf")

# ================== MEMORY INIT ==================
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# ================== PROCESS PDF ==================
def process_pdf(uploaded_file, filename):

    with open(filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader(filename)
    docs = loader.load()

    return docs

# ================== PROMPT ==================
prompt = PromptTemplate.from_template(
"""
You are an AI ATS system and career assistant.

Previous conversation:
{chat_history}

Context from resume & job description:
{context}

User question:
{question}

Give helpful professional answer.
"""
)

# ================== BUILD VECTOR DB ==================
if resume_file and jd_file:

    if st.sidebar.button("Analyze Resume"):

        st.info("Processing...")

        resume_docs = process_pdf(resume_file, "resume.pdf")
        jd_docs = process_pdf(jd_file, "jd.pdf")

        all_docs = resume_docs + jd_docs

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(all_docs)

        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.from_documents(chunks, embedding)

        st.session_state.vdb = vector_db

        st.success("✅ ATS Ready! Chat below 👇")

# ================== RAG + MEMORY FUNCTION ==================
def ask_llm(question):

    retriever = st.session_state.vdb.as_retriever(search_kwargs={"k":4})

    docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in docs])

    # LOAD MEMORY
    chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]

    chain = prompt | llm | parser

    result = chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": chat_history
        }
    )

    # SAVE MEMORY
    st.session_state.memory.save_context(
        {"input": question},
        {"output": result}
    )

    return result

# ================== CHAT UI ==================
if "vdb" in st.session_state:

    st.subheader("📊 ATS Analysis")

    if st.button("Generate ATS Report"):
        report = ask_llm("Give full ATS analysis of this resume")
        st.write(report)

    st.subheader("💬 Chat with Resume (Memory Enabled)")

    user_q = st.text_input("Ask anything")

    if user_q:
        response = ask_llm(user_q)
        st.write(response)

else:
    st.warning("⚠ Upload Resume + JD and click Analyze")