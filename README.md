# 🚀 ATS Resume Analyzer with Memory (RAG + LLM)

An **AI-powered ATS (Applicant Tracking System) Resume Analyzer** that evaluates resumes against job descriptions using **LLMs, Retrieval-Augmented Generation (RAG), vector databases, and conversation memory**.

The application allows users to upload a **Resume (PDF)** and **Job Description (PDF)**, generate a **complete ATS analysis**, and **chat interactively with the system** to improve their resume.

This project demonstrates how **AI-powered resume analysis systems can assist candidates in optimizing resumes for ATS systems used by recruiters.**

---

# 🚀 Live Demo

Try the application here:

🔗 *Add your Streamlit deployment link here*

---

# ✨ Features

- Upload **Resume PDF**
- Upload **Job Description PDF**
- **AI-powered ATS resume analysis**
- **Resume vs Job Description comparison**
- **Skill gap identification**
- **Suggestions to improve resume**
- **Interactive chat with resume context**
- **Conversation memory for contextual responses**
- Fast inference using **Groq LLM**
- **Vector search using FAISS**

---

# 🧠 Architecture

The application uses a **RAG (Retrieval-Augmented Generation) pipeline**:


Resume + Job Description Upload
↓
PDF Processing (PyPDFLoader)
↓
Document Chunking
(RecursiveCharacterTextSplitter)
↓
Embeddings Generation
(HuggingFace Embeddings)
↓
Vector Database
(FAISS)
↓
Retriever
↓
LLM (Groq - Llama 3.3)
↓
ATS Analysis + Chat Response


Conversation memory allows the AI to **remember previous questions and maintain context during the chat.**

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Streamlit | Web interface |
| LangChain | LLM orchestration |
| Groq LLM | High-speed LLM inference |
| FAISS | Vector database for semantic search |
| HuggingFace Embeddings | Text embeddings |
| PyPDFLoader | PDF document loading |
| RecursiveCharacterTextSplitter | Document chunking |
| ConversationBufferMemory | Chat memory |

---

# 📁 Project Structure


ATS-Resume-Analyzer
│
├── app.py
├── requirements.txt
├── README.md
├── .env


---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ATS-Resume-Analyzer.git

Navigate to the project folder:

cd ATS-Resume-Analyzer

Install dependencies:

pip install -r requirements.txt
🔑 Setup Environment Variables

Create a .env file and add your Groq API key:

GROQ_API_KEY=your_api_key_here

Get the API key from:

https://console.groq.com

▶️ Run the Application

Start the Streamlit app:

streamlit run app.py

The application will start at:

http://localhost:8501