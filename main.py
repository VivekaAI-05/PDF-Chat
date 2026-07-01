import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import tempfile
from dotenv import load_dotenv
import os

load_dotenv()
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ─── Config ───────────────────────────────────────────────────────────────────
# openrouter.ai -ல் இலவசமாக key பெறுங்கள்


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ─── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot Using RAG")

# ─── Session state init ───────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── PDF Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("PDF processing..."):
        pdf_bytes = uploaded_file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)

    st.success("✅ PDF processed successfully , Ask questions about it.")

# ─── Chat ─────────────────────────────────────────────────────────────────────
if st.session_state.vectorstore is not None:

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("❓Ask Question About the PDF...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("Assistant"):
            with st.spinner("Thinking..."):

                # RAG: தொடர்புடைய chunks எடு
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )
                relevant_docs = retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                prompt = f"""Answer the question based on the context below only.
If the answer is not in the context, say "This PDF does not contain the information you're looking for".

Context:
{context}

Question: {question}"""

                response = client.chat.completions.create(
                    model="openrouter/free", # இலவச model
                messages=[
                        {"role": "user", "content": prompt}
                    ],
                )

                answer = response.choices[0].message.content
                st.write(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

    if st.session_state.chat_history:
        if st.button("🗑️ Chat Erase"):
            st.session_state.chat_history = []
            st.rerun()


