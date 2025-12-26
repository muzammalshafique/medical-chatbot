import gradio as gr
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize embeddings and vector store
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

# Store conversation histories (key: session_hash from Gradio)
conversation_histories = {}

def chat_with_memory(message, history, session_hash):
    """
    Chat function with conversation memory
    
    Args:
        message: Current user message
        history: Gradio chat history format [[user_msg, bot_msg], ...]
        session_hash: Unique session identifier from Gradio
    """
    # Initialize history for new sessions
    if session_hash not in conversation_histories:
        conversation_histories[session_hash] = []
    
    chat_history = conversation_histories[session_hash]
    
    # Build conversation context from stored history
    history_context = ""
    if chat_history:
        history_context = "\n\nPrevious conversation context:\n"
        for human_msg, ai_msg in chat_history[-5:]:  # Last 5 exchanges
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
        print(f"Session {session_hash[:8]} - User: {message}")
        response = rag_chain.invoke({"input": message})
        answer = response["answer"]
        print(f"Session {session_hash[:8]} - Bot: {answer[:100]}...")
        
        # Store conversation in history
        chat_history.append((message, answer))
        
        # Keep only last 10 exchanges
        if len(chat_history) > 10:
            chat_history.pop(0)
        
        return answer
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        print(f"Error: {error_msg}")
        return error_msg

def clear_history(session_hash):
    """Clear conversation history for current session"""
    if session_hash in conversation_histories:
        conversation_histories[session_hash] = []
        print(f"Cleared history for session {session_hash[:8]}")
    return None  # Clear the chat interface

# Quick question buttons
quick_questions = [
    "What is a brain aneurysm?",
    "What are the symptoms of a brain aneurysm?",
    "What causes brain aneurysms?",
    "How is a brain aneurysm diagnosed?",
    "What are the treatment options for brain aneurysm?",
    "What are the risk factors for brain aneurysm?",
    "Can brain aneurysms be prevented?",
    "What is the recovery time after brain aneurysm surgery?"
]

# Create Gradio Interface
with gr.Blocks(
    title="Brain Aneurysm Medical Assistant",
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="indigo",
    ),
    css="""
    .gradio-container {
        max-width: 900px !important;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        color: #856404;
    }
    """
) as demo:
    
    # Store session hash in state
    session_state = gr.State(value=None)
    
    # Header
    gr.Markdown(
        """
        # 🧠 Brain Aneurysm Medical Assistant
        ### AI-powered healthcare information and support
        """
    )
    
    # Disclaimer
    gr.HTML(
        """
        <div class="disclaimer">
            ⚕️ <strong>Medical Disclaimer:</strong> This chatbot provides general information only and is not a substitute for professional medical advice, diagnosis, or treatment.
        </div>
        """
    )
    
    # Chat interface
    chatbot = gr.Chatbot(
        label="Chat with Medical Assistant",
        height=500,
        avatar_images=(None, "🩺")
    )
    
    # Message input
    msg = gr.Textbox(
        label="Your Question",
        placeholder="Type your question here...",
        lines=2,
        max_lines=4
    )
    
    # Buttons row
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary", scale=2)
        clear_btn = gr.Button("Clear Chat", scale=1)
    
    # Quick questions
    gr.Markdown("### 💡 Quick Questions")
    with gr.Row():
        quick_btns = []
        for i in range(0, len(quick_questions), 2):
            with gr.Column():
                quick_btns.append(gr.Button(quick_questions[i], size="sm"))
                if i + 1 < len(quick_questions):
                    quick_btns.append(gr.Button(quick_questions[i + 1], size="sm"))
    
    # Initialize session on load
    def init_session():
        import uuid
        return str(uuid.uuid4())
    
    # Set up event handlers
    def user_message(user_msg, history, session_hash):
        """Handle user message submission"""
        if not session_hash:
            session_hash = init_session()
        return "", history + [[user_msg, None]], session_hash
    
    def bot_response(history, session_hash):
        """Generate bot response"""
        user_msg = history[-1][0]
        bot_msg = chat_with_memory(user_msg, history, session_hash)
        history[-1][1] = bot_msg
        return history
    
    def quick_question_click(question, history, session_hash):
        """Handle quick question button click"""
        if not session_hash:
            session_hash = init_session()
        history = history + [[question, None]]
        bot_msg = chat_with_memory(question, history, session_hash)
        history[-1][1] = bot_msg
        return history, session_hash
    
    def clear_chat(session_hash):
        """Clear chat interface and history"""
        if session_hash:
            clear_history(session_hash)
        return None, None
    
    # Initialize session on load
    demo.load(init_session, None, session_state)
    
    # Submit message
    msg.submit(
        user_message,
        [msg, chatbot, session_state],
        [msg, chatbot, session_state],
        queue=False
    ).then(
        bot_response,
        [chatbot, session_state],
        chatbot
    )
    
    submit_btn.click(
        user_message,
        [msg, chatbot, session_state],
        [msg, chatbot, session_state],
        queue=False
    ).then(
        bot_response,
        [chatbot, session_state],
        chatbot
    )
    
    # Quick question buttons
    for btn in quick_btns:
        btn.click(
            quick_question_click,
            [btn, chatbot, session_state],
            [chatbot, session_state]
        )
    
    # Clear button
    clear_btn.click(
        clear_chat,
        session_state,
        [chatbot, session_state]
    )

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Medical Chatbot (Gradio) Starting...")
    print("="*50)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )