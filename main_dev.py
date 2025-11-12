import streamlit as st
import os, base64, time, json
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import re

# ========== ENV SETUP ==========
load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GOOGLE_GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL")
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

@st.cache_resource
def load_model():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# @st.cache_resource
# def load_vectorstore(embeddings):
#     return Chroma(
#         persist_directory="./chroma_db/chroma_files_db",
#         embedding_function=embeddings
#     )

VECTORSTORE_CACHE = {"store": None, "last_update": 0}

def load_vectorstore(embeddings, persist_directory="./chroma_db/chroma_files_db"):
    """
    Load Chroma vector store and auto-reload if DB files are updated.
    """
    # Get the latest modification timestamp in the directory
    latest_mod = max(
        (os.path.getmtime(os.path.join(root, f))
         for root, _, files in os.walk(persist_directory) for f in files),
        default=0
    )

    # Reload if never loaded or directory updated
    if VECTORSTORE_CACHE["store"] is None or latest_mod > VECTORSTORE_CACHE["last_update"]:
        VECTORSTORE_CACHE["store"] = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        VECTORSTORE_CACHE["last_update"] = latest_mod

    return VECTORSTORE_CACHE["store"]

@st.cache_data
def img_to_base64(path, size=(100, 100)):
    img = Image.open(path)
    img = img.resize(size)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

model = load_model()
embeddings = load_embeddings()
file_vector_store = load_vectorstore(embeddings)

# ========== RETRIEVAL TOOL ==========
@tool(response_format="content")
def retrieve_file_context(query: str):
    """Retrieve information from local Excel, PDF, and text files."""
    retrieved_docs = file_vector_store.similarity_search(query, k=10)
    if not retrieved_docs:
        return "No relevant content found."
    serialized = "\n\n".join(
        f"📄 **Source:** {doc.metadata}\n\n{doc.page_content[:800]}"
        for doc in retrieved_docs
    )
    return serialized

def is_valid_json(data: str) -> bool:
    try:
        json.loads(data)
        return True
    except (ValueError, TypeError):
        return False

def safe_json_loads(text):
    """
    Tries to repair and parse invalid JSON-like strings often returned by LLMs.
    Converts single quotes to double quotes, escapes inner apostrophes, and
    removes trailing commas.
    """
    fixed = text.strip()

    # Fix Python-style single quotes and nested apostrophes like The Leader's
    # Replace only the outer single quotes with double quotes
    fixed = fixed.replace("'", '"')

    # Remove trailing commas before brackets/braces
    fixed = re.sub(r",\s*}", "}", fixed)
    fixed = re.sub(r",\s*]", "]", fixed)

    # Clean doubled quotes
    fixed = re.sub(r'""', '"', fixed)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        # Fallback: attempt literal_eval (handles Python-like structures)
        import ast
        try:
            return ast.literal_eval(text)
        except Exception:
            return {"raw_text": text}

def extract_ai_message(response):
    """Extracts plain text safely from any response object."""
    if hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)

    if is_valid_json(content):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list) and "text" in parsed[0]:
                return parsed[0]["text"]
            return json.dumps(parsed, indent=2)
        except Exception:
            return content
    return content

# ========== STREAMLIT UI CONFIG ==========
st.set_page_config(
    page_title="CHA, CDO HR Assistant",
    page_icon=img_to_base64("assets/logo/CDO Logo.png")
)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Call the function early in your app
load_css("assets/style.css")

# ========== SESSION STATE ==========
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    system_prompt = (
        "You are CHA, a reliable HR Assistant for CDO. "
        "You have access to a retrieval tool for company documents (PDF, Excel, and TXT). "
        "Use the tool to find relevant information and provide concise, factual answers. "
        "If appropriate, ask the user about other data or information related to your current context."
        "If no relevant content is found, respond politely using general HR knowledge."
    )
    st.session_state.agent = create_agent(model, [retrieve_file_context], system_prompt=system_prompt)

agent = st.session_state.agent

# ================== AUTHENTICATION ==================
# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Get current user
user = st.user

# Authentication flow
if not user or not user.get("is_logged_in", False):
    st.title("Authentication Module 🔒")
    st.write("Please login with your Google account to access CHA.")

    if st.button("Authenticate with Google"):
        st.login("google")
        st.session_state.authenticated = True

    st.stop()  # stop execution until user logs in

st.badge("Online", icon=":material/check:", color="green")
st.markdown(
    """
    <h1 style='margin-top: 10px; margin-bottom: 0; font-family: "Segoe UI", sans-serif;'>
        Welcome back!
    </h1>
    <h4 style='font-family: "Segoe UI", sans-serif;'>
        I’m Cha, your CDO HR Assistant. How may I assist you? ✨
    </h4>
    """,
    unsafe_allow_html=True
)

# ========== AVATARS ==========
USER_AVATAR = f"{user['picture']}"
BOT_AVATAR = "https://png.pngtree.com/png-vector/20230225/ourmid/pngtree-smart-chatbot-cartoon-clipart-png-image_6620453.png"

# ========== DISPLAY CHAT HISTORY ==========
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(f"<div class='chat-message-user'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(f"<div class='chat-message-bot'>{msg['content']}</div>", unsafe_allow_html=True)

# ========== HANDLE CHAT INPUT ==========
if prompt := st.chat_input(placeholder="Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(f"<div class='chat-message-user'>{prompt}</div>", unsafe_allow_html=True)

    # AI response container
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        with st.spinner("Assistant is thinking..."):

            # Stream model response
            final_response = None
            for step in agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                stream_mode="values",
            ):
                final_response = step["messages"][-1]

            # final_response = "The Leaders Playbook is a toolkit for leaders to manage employee experience at CDO. It covers various aspects of the employee journey, including attracting, onboarding, engaging, developing, and offboarding employees. The playbook emphasizes leading teams with passion, understanding, and vigor, treating employees as an extension of the company's family, and focusing on both results and employee actions and behaviors for success. It also includes frameworks, processes, policies, and tools for each stage of the employee experience, such as performance management, talent development, and individual development plans."

            # Extract AI text safely
            ai_message = extract_ai_message(final_response)
            ai_message = str(extract_ai_message(final_response))
            validate_message = safe_json_loads(ai_message)

            if "raw_text" in validate_message:
                ai_message = validate_message["raw_text"]
            else:
                ai_message = validate_message
                if isinstance(ai_message, list):
                    ai_message = ai_message[0].get("text", str(ai_message[0]))
                elif isinstance(ai_message, dict):
                    ai_message = ai_message.get("text", str(ai_message))
                else:
                    ai_message = str(ai_message)

            # Simulate typing effect asynchronously
            typed_text = ""
            for char in ai_message:
                typed_text += char
                message_placeholder.markdown(typed_text + "▌")
                # message_placeholder.markdown(f"<div class='chat-message-bot'>{typed_text} ▌</div>", unsafe_allow_html=True)
                time.sleep(0.005)

            # message_placeholder.markdown(ai_message)
            message_placeholder.markdown(f"<div class='chat-message-bot'>{ai_message}</div>", unsafe_allow_html=True)

    # Save conversation
    st.session_state.messages.append({"role": "assistant", "content": ai_message})




