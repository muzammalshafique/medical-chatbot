import os
import streamlit as st
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_embeddings
from src.prompt import system_prompt

import base64

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BRAIN_BG = img_to_base64("static/images/brain_aneurysm.png")
ICON_BG = img_to_base64("static/images/icon.png")

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Brain Aneurysm Medical Assistant",
    page_icon="🧠",
    layout="centered"
)

# =====================
# ENV
# =====================
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =====================
# SESSION STATE
# =====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================
# CSS
# =====================
st.markdown(f"""
<style>

/* Hide Streamlit branding */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Full page background gradient */
.stApp {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}}

/* Remove default padding */
.block-container {{
    padding: 2rem 1rem;
    max-width: 700px;
}}

/* Hide default streamlit elements */
.element-container {{
    margin-bottom: 0 !important;
}}

/* Main container */
.main-container {{
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    overflow: hidden;
}}

/* Header section */
.header {{
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 22px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.header-left {{
    display: flex;
    align-items: center;
    gap: 14px;
}}

.header-icon {{
    width: 48px;
    height: 48px;
    background: white;
    border-radius: 50%;
    padding: 6px;
}}

.header-text h1 {{
    font-size: 19px;
    margin: 0;
    font-weight: 600;
    color: white;
}}

.header-text p {{
    font-size: 12px;
    opacity: 0.95;
    margin: 3px 0 0 0;
    color: white;
}}

/* Disclaimer */
.disclaimer {{
    background: #fff9e6;
    border-left: 4px solid #ffc107;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 16px;
    font-size: 13px;
    color: #856404;
    line-height: 1.5;
}}

/* Chat area */
.chat-area {{
    min-height: 400px;
    max-height: 400px;
    overflow-y: auto;
    padding: 25px;
    background: 
        linear-gradient(rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.92)),
        url("data:image/png;base64,{BRAIN_BG}");
    background-repeat: no-repeat;
    background-position: center center;
    background-size: 350px auto;
}}

.chat-area::-webkit-scrollbar {{
    width: 6px;
}}

.chat-area::-webkit-scrollbar-thumb {{
    background: #ccc;
    border-radius: 3px;
}}

/* Welcome message */
.welcome {{
    text-align: center;
    padding: 80px 30px;
}}

.welcome h2 {{
    font-size: 20px;
    color: #333;
    margin-bottom: 12px;
}}

.welcome p {{
    font-size: 14px;
    color: #666;
    line-height: 1.6;
}}

/* Chat messages */
.msg {{
    margin-bottom: 16px;
    display: flex;
}}

.msg.user {{
    justify-content: flex-end;
}}

.msg.bot {{
    justify-content: flex-start;
}}

.bubble {{
    max-width: 68%;
    padding: 11px 16px;
    border-radius: 16px;
    font-size: 14px;
    line-height: 1.5;
}}

.user .bubble {{
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-bottom-right-radius: 4px;
}}

.bot .bubble {{
    background: #f0f0f0;
    color: #333;
    border-bottom-left-radius: 4px;
}}

/* Quick buttons area */
.quick-area {{
    padding: 14px 16px;
    border-top: 1px solid #e5e5e5;
    overflow-x: auto;
    overflow-y: hidden;
    white-space: nowrap;
    display: flex;
    gap: 8px;
}}

.quick-area::-webkit-scrollbar {{
    height: 5px;
}}

.quick-area::-webkit-scrollbar-thumb {{
    background: #bbb;
    border-radius: 3px;
}}

/* Input area */
.input-area {{
    padding: 16px;
    border-top: 1px solid #e5e5e5;
    background: #fafafa;
}}

/* Streamlit button overrides */
div[data-testid="column"] {{
    padding: 0 !important;
}}

.stButton {{
    margin: 0;
}}

.stButton button {{
    width: 100%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    padding: 9px 18px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}}

.stButton button:hover {{
    opacity: 0.92;
    transform: translateY(-1px);
}}

/* Clear button style */
.clear-btn button {{
    background: rgba(255,255,255,0.25) !important;
    border: 1px solid rgba(255,255,255,0.4) !important;
    padding: 7px 16px !important;
    font-size: 12px !important;
}}

.clear-btn button:hover {{
    background: rgba(255,255,255,0.35) !important;
}}

/* Send button */
.send-btn button {{
    border-radius: 50% !important;
    width: 46px !important;
    height: 46px !important;
    padding: 0 !important;
    font-size: 20px !important;
    min-width: 46px !important;
}}

/* Input field */
.stTextInput input {{
    border-radius: 24px;
    border: 1px solid #ddd;
    padding: 12px 20px;
    font-size: 14px;
    background: white;
}}

.stTextInput input:focus {{
    border-color: #667eea;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.15);
}}

</style>
""", unsafe_allow_html=True)

# =====================
# VECTOR STORE (CACHE)
# =====================
@st.cache_resource
def load_vectorstore():
    embeddings = download_embeddings()
    return PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot",
        embedding=embeddings
    )

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    model="groq/compound",
    groq_api_key=GROQ_API_KEY,
    temperature=0.7
)

# =====================
# RAG FUNCTION
# =====================
def answer_question(query):
    history_context = ""
    for u, a in st.session_state.chat_history[-5:]:
        history_context += f"User: {u}\nAssistant: {a}\n\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", history_context + "\nCurrent question: {input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    response = rag_chain.invoke({"input": query})
    return response["answer"]

# =====================
# UI START
# =====================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# HEADER
col_left, col_right = st.columns([5, 1])

with col_left:
    st.markdown(f"""
    <div class="header">
        <div class="header-left">
            <img class="header-icon" src="data:image/png;base64,{ICON_BG}">
            <div class="header-text">
                <h1>Brain Aneurysm Medical Assistant</h1>
                <p>AI-powered healthcare information and support</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div style="position: absolute; top: 32px; right: 28px; z-index: 100;" class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# DISCLAIMER
st.markdown("""
<div class="disclaimer">
💡 <strong>Medical Disclaimer:</strong> This chatbot provides general information only and is not a substitute for professional medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)

# CHAT AREA
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if len(st.session_state.chat_history) == 0:
    st.markdown("""
    <div class="welcome">
        <h2>👋 Welcome to Your Brain Aneurysm Medical Assistant</h2>
        <p>I'm here to help answer your questions about brain aneurysms. Feel free to ask me anything, and I'll do my best to provide accurate and helpful information based on medical knowledge.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for role, msg in st.session_state.chat_history:
        st.markdown(f"""
        <div class="msg {role}">
            <div class="bubble">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# QUICK QUESTIONS
st.markdown('<div class="quick-area">', unsafe_allow_html=True)

questions = [
    "What is a brain aneurysm?",
    "What are the symptoms?",
    "What causes it?",
    "How is it diagnosed?",
    "Treatment options?",
    "Risk factors?",
    "Can it be prevented?",
    "Recovery time?"
]

cols = st.columns(len(questions))
for i, q in enumerate(questions):
    with cols[i]:
        if st.button(q, key=f"q{i}"):
            ans = answer_question(q)
            st.session_state.chat_history.append(("user", q))
            st.session_state.chat_history.append(("bot", ans))
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# INPUT AREA
st.markdown('<div class="input-area">', unsafe_allow_html=True)

c1, c2 = st.columns([6, 1])

with c1:
    query = st.text_input("Type your question here...", label_visibility="collapsed", key="input")

with c2:
    st.markdown('<div class="send-btn">', unsafe_allow_html=True)
    if st.button("➤"):
        if query:
            ans = answer_question(query)
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("bot", ans))
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)