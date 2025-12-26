from flask import Flask, render_template, jsonify, request, session
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
# from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import uuid
from langchain_huggingface import HuggingFaceEndpoint

app = Flask(__name__)

app.secret_key = "599a121a760a3b21d3e14af8bf1690397a9963976c522ec4f049284854ec0c19"

load_dotenv()

PINECONE_API_KEY = "pcsk_2YPrcL_Kb5BAb8bbDqS5USWqVa4sAFnDZCReX2rZKM956amvfrvfHvtoxXvzUCB3c5nE82"
GROQ_API_KEY="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"

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# chatModel = ChatOllama(
#     model="llama3",
# )

chatModel = ChatGroq(
    model="groq/compound",
    groq_api_key=GROQ_API_KEY,
    temperature=0.7
)

# chatModel = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
#     task="conversational",
# )

# Store conversation histories for each session
conversation_histories = {}

@app.route("/")
def index():
    # Create unique session ID for each user
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    session_id = session.get('session_id')
    
    # Initialize conversation history for new sessions
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    chat_history = conversation_histories[session_id]
    
    # Build conversation context from history
    history_context = ""
    if chat_history:
        history_context = "\n\nPrevious conversation context:\n"
        # Include last 5 exchanges for context
        for human_msg, ai_msg in chat_history[-5:]:
            history_context += f"User: {human_msg}\nAssistant: {ai_msg}\n\n"
    
    # Create prompt with conversation history
    prompt_with_memory = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nYou have access to previous conversation context. Use it to provide relevant and contextual responses. If the user asks follow-up questions, refer back to the previous conversation."),
        ("human", f"{history_context}Current question: {{input}}")
    ])
    
    # Create chains with memory-enhanced prompt
    question_answer_chain = create_stuff_documents_chain(chatModel, prompt_with_memory)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Get response
    print(f"Session {session_id} - User: {msg}")
    response = rag_chain.invoke({"input": msg})
    answer = response["answer"]
    print(f"Session {session_id} - Bot: {answer}")
    
    # Store conversation in history
    chat_history.append((msg, answer))
    
    # Keep only last 10 exchanges to prevent memory overflow
    if len(chat_history) > 10:
        chat_history.pop(0)
    
    print(f"Session {session_id} - History size: {len(chat_history)} exchanges")
    
    return str(answer)


@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear conversation history for current session"""
    session_id = session.get('session_id')
    if session_id in conversation_histories:
        conversation_histories[session_id] = []
        print(f"Cleared history for session {session_id}")
    return jsonify({"status": "success", "message": "Conversation history cleared"})


@app.route("/history", methods=["GET"])
def get_history():
    """Get conversation history - useful for debugging"""
    session_id = session.get('session_id')
    if session_id in conversation_histories:
        history = conversation_histories[session_id]
        return jsonify({
            "session_id": session_id,
            "history_length": len(history),
            "history": history
        })
    return jsonify({"session_id": session_id, "history_length": 0, "history": []})


if __name__ == '__main__':
    app.run()
    # host="0.0.0.0", port=8080, debug=True
