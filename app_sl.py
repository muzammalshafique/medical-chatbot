import streamlit as st
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import uuid

# Page configuration
st.set_page_config(
    page_title="Brain Aneurysm Medical Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        max-width: 900px;
        margin: 0 auto;
    }
    .stAlert {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 12px;
        color: #856404;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #667eea;
        color: white;
        align-items: flex-end;
    }
    .chat-message.bot {
        background-color: #f0f2f6;
        color: #333;
    }
    .chat-message .avatar {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .quick-question-btn {
        margin: 0.25rem;
    }
    div[data-testid="stButton"] > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Initialize models (cached to avoid reloading)
@st.cache_resource
def initialize_models():
    """Initialize embeddings, vector store, and chat model"""
    print("Loading embeddings...")
    embeddings = download_embeddings()
    
    print("Connecting to Pinecone...")
    index_name = "medical-chatbot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    print("Initializing ChatGroq...")
    chatModel = ChatGroq(
        model="groq/compound",
        groq_api_key=GROQ_API_KEY,
        temperature=0.7
    )
    
    return retriever, chatModel

# Load models
retriever, chatModel = initialize_models()

def get_response(message):
    """Get response from the chatbot with conversation memory"""
    
    # Build conversation context from history
    history_context = ""
    if st.session_state.conversation_history:
        history_context = "\n\nPrevious conversation context:\n"
        # Include last 5 exchanges for context
        for human_msg, ai_msg in st.session_state.conversation_history[-5:]:
            history_context += f"User: {human_msg}\nAssistant: {ai_msg}\n\n"
    
    # Create prompt with conversation history
    prompt_with_memory = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nYou have access to previous conversation context. Use it to provide relevant and contextual responses. If the user asks follow-up questions, refer back to the previous conversation."),
        ("human", f"{history_context}Current question: {{input}}")
    ])
    
    # Create chains with memory-enhanced prompt
    question_answer_chain = create_stuff_documents_chain(chatModel, prompt_with_memory)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    try:
        # Get response
        print(f"Session {st.session_state.session_id[:8]} - User: {message}")
        response = rag_chain.invoke({"input": message})
        answer = response["answer"]
        print(f"Session {st.session_state.session_id[:8]} - Bot: {answer[:100]}...")
        
        # Store conversation in history
        st.session_state.conversation_history.append((message, answer))
        
        # Keep only last 10 exchanges
        if len(st.session_state.conversation_history) > 10:
            st.session_state.conversation_history.pop(0)
        
        return answer
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        print(f"Error: {error_msg}")
        return error_msg

def clear_chat():
    """Clear chat history"""
    st.session_state.messages = []
    st.session_state.conversation_history = []
    st.session_state.session_id = str(uuid.uuid4())
    print(f"Cleared history, new session: {st.session_state.session_id[:8]}")

# Sidebar
with st.sidebar:
    st.markdown("### 🧠 Brain Aneurysm Assistant")
    st.markdown("---")
    
    st.markdown("### 💡 Quick Questions")
    
    quick_questions = [
        "What is a brain aneurysm?",
        "What are the symptoms?",
        "What causes it?",
        "How is it diagnosed?",
        "Treatment options?",
        "Risk factors?",
        "Can it be prevented?",
        "Recovery time?"
    ]
    
    for question in quick_questions:
        if st.button(question, key=f"quick_{question}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                response = get_response(question)
                st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        clear_chat()
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    st.markdown(f"**History:** {len(st.session_state.conversation_history)} exchanges")

# Main content
st.title("🧠 Brain Aneurysm Medical Assistant")
st.markdown("### AI-powered healthcare information and support")

# Disclaimer
st.warning("⚕️ **Medical Disclaimer:** This chatbot provides general information only and is not a substitute for professional medical advice, diagnosis, or treatment.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🩺"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Thinking..."):
            response = get_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})

# Welcome message if no messages
if len(st.session_state.messages) == 0:
    st.markdown("---")
    st.markdown("""
    ### 👋 Welcome to Your Medical Assistant
    
    I'm here to help answer your questions about brain aneurysms. Feel free to ask me anything, and I'll do my best to provide accurate and helpful information based on medical knowledge.
    
    You can:
    - Type your question in the chat box below
    - Click on quick questions in the sidebar
    - Ask follow-up questions (I remember our conversation!)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "Powered by LangChain, Groq, and Pinecone | "
    "Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)