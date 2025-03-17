import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Assistant | LangChain + Gemma 2B",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for Glassmorphism & Professional Look
st.markdown(
    """
    <style>
        /* Main container */
        .stApp {
            background-color: #0d1117;
            color: white;
        }

        /* Header */
        .header-title {
            text-align: center;
            font-size: 38px;
            font-weight: bold;
            color: #58a6ff;
        }

        /* Subheading */
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #8b949e;
        }

        /* Chat container */
        .chat-box {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            padding: 15px;
            border-radius: 10px;
            color: white;
            font-size: 18px;
            font-family: 'Arial', sans-serif;
            box-shadow: 2px 2px 10px rgba(255,255,255,0.1);
        }

        /* Input field */
        .stTextInput > label {
            font-size: 18px;
            font-weight: bold;
            color: #c9d1d9;
        }

        /* Footer */
        .footer {
            text-align: center;
            font-size: 14px;
            color: #8b949e;
            margin-top: 20px;
        }

        /* Sidebar */
        .sidebar-content {
            background: #161b22;
            padding: 15px;
            border-radius: 10px;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Header
st.markdown('<h1 class="header-title">🤖 AI Chat Assistant - LangChain + Gemma 2B</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Smart, Adaptive & Context-Aware AI Chatbot</p>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar for Model Selection & Info
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/LangChain_logo.svg/512px-LangChain_logo.svg.png", width=160)
    st.header("⚙️ Model Configuration")
    
    # Model Selection
    model_choice = st.selectbox(
        "Select AI Model",
        ["gemma:2b", "mistral:7b", "llama3:8b"],
        help="Choose the LLM model for generating responses"
    )

    st.markdown("**📝 Project Details:**")
    st.write("- Built with **LangChain**, **Streamlit**, and **Ollama**")
    st.write("- Supports **Conversational Memory**")
    st.write("- Created by **[Shivam Kumar](https://www.linkedin.com/in/shivamlko9832/)**")

# Initialize LangChain Components
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a highly intelligent AI assistant. Provide precise and insightful answers."),
        ("user", "Question: {question}"),
    ]
)

llm = Ollama(model=model_choice)  # Dynamic Model Selection
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Session State for Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
input_text = st.text_input("💡 Ask Anything:", placeholder="E.g., Explain self-supervised learning.", help="Type your query and hit Enter")

# Processing & Memory Handling
if input_text:
    with st.spinner("🤖 Thinking..."):
        response = chain.invoke({"question": input_text})

    # Store the conversation in session memory
    st.session_state.chat_history.append((input_text, response))

# Display Chat History
st.markdown("### 🗂️ Chat History")
for q, r in reversed(st.session_state.chat_history):
    st.markdown(f"**🟢 You:** {q}")
    st.markdown(f'<div class="chat-box">🤖 **AI:** {r}</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <hr>
    <p class="footer">🚀 Built with <b>LangChain</b>, <b>Streamlit</b>, and <b>Ollama</b>. 
    <br> Developed by <a href="https://www.linkedin.com/in/shivamlko9832" target="_blank">Shivam Kumar</a>.</p>
    """,
    unsafe_allow_html=True
)
