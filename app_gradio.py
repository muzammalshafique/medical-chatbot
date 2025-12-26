import os
import gradio as gr
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_embeddings
from src.prompt import system_prompt

# ======================
# ENV
# ======================
load_dotenv()

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ======================
# VECTOR STORE
# ======================
embeddings = download_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ======================
# LLM
# ======================
llm = ChatGroq(
    model="groq/compound",
    groq_api_key=GROQ_API_KEY,
    temperature=0.7
)

# ======================
# RAG FUNCTION
# ======================
def chat_fn(message, history):

    history_context = ""
    for u, a in history[-5:]:
        history_context += f"User: {u}\nAssistant: {a}\n\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", history_context + "\nCurrent question: {input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    response = rag_chain.invoke({"input": message})
    answer = response["answer"]

    history.append((message, answer))
    return history, history


# ======================
# CSS (FROM YOUR UI)
# ======================
CUSTOM_CSS = """
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.chat-wrapper {
    max-width: 900px;
    height: 90vh;
    margin: auto;
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.header {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 25px;
    display: flex;
    align-items: center;
    gap: 20px;
}

.header img {
    width: 60px;
    border-radius: 50%;
    background: white;
    padding: 6px;
}

.header h1 {
    font-size: 24px;
    margin: 0;
}

.header p {
    font-size: 14px;
    opacity: 0.95;
}

.disclaimer {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 12px;
    padding: 14px;
    margin: 16px;
    font-size: 13px;
    color: #856404;
    text-align: center;
}

.quick-buttons button {
    border-radius: 20px !important;
    border: 2px solid #e9ecef !important;
    background: #f8f9fa !important;
    font-size: 14px !important;
}

.quick-buttons button:hover {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}

.gr-chatbot {
    background: #f8f9fa;
}
"""


# ======================
# UI
# ======================
with gr.Blocks(css=CUSTOM_CSS, title="Brain Aneurysm Medical Assistant") as demo:

    with gr.Column(elem_classes="chat-wrapper"):

        # HEADER
        with gr.Row(elem_classes="header"):
            gr.Image("static/images/icon.png", show_label=False)
            gr.Markdown(
                """
                ### Brain Aneurysm Medical Assistant  
                AI-powered healthcare information and support
                """
            )
            clear_btn = gr.Button("🗑️ Clear Chat")

        # DISCLAIMER
        gr.Markdown(
            """
            ⚕️ **Medical Disclaimer:**  
            This chatbot provides general information only and is not a substitute for professional medical advice.
            """,
            elem_classes="disclaimer"
        )

        # CHAT
        chatbot = gr.Chatbot(
            height=420,
            avatar_images=("👤", "🩺")
        )

        state = gr.State([])

        # QUICK QUESTIONS
        with gr.Row(elem_classes="quick-buttons"):
            quick_qs = [
                "What is a brain aneurysm?",
                "What are the symptoms?",
                "What causes brain aneurysms?",
                "How is it diagnosed?",
                "Treatment options?",
                "Risk factors?",
                "Can it be prevented?",
                "Recovery time after surgery?"
            ]

            for q in quick_qs:
                gr.Button(q).click(
                    lambda x=q: x,
                    outputs=None
                ).then(
                    chat_fn,
                    inputs=[gr.State(q), state],
                    outputs=[chatbot, state]
                )

        # INPUT
        with gr.Row():
            txt = gr.Textbox(
                placeholder="Type your question here...",
                show_label=False,
                scale=4
            )
            send = gr.Button("➤", scale=1)

        send.click(chat_fn, [txt, state], [chatbot, state])
        txt.submit(chat_fn, [txt, state], [chatbot, state])

        clear_btn.click(
            lambda: ([], []),
            None,
            [chatbot, state]
        )

demo.launch()