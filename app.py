import streamlit as st
import google.generativeai as genai
import os
import json
import uuid
import datetime
import hashlib
import extra_streamlit_components as stx
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("Secrets not found! Please create .streamlit/secrets.toml")
    st.stop()
    
HISTORY_FILE = "history.json" # Kept for backward compatibility or migration if needed
USERS_FILE = "users.json"

# Configure Gemini
genai.configure(api_key=API_KEY)

# Generator Config
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

# Page Config
st.set_page_config(
    page_title="Office Brain AI",
    page_icon="🧠",
    layout="wide"
)

# --- HELPER FUNCTIONS ---

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def load_history(username=None):
    # If Guest, return empty dict (handled in session state usually, but good fallback)
    if not username:
        return {}
    
    filename = f"history_{username}.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_history(history, username=None):
    if not username:
        return # Guest mode, do not save to file
        
    filename = f"history_{username}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

@st.cache_resource
def load_faiss_index():
    # Check if index exists, if not, create it
    if not os.path.exists("faiss_index"):
        if os.path.exists("knowledge.txt"):
            # Generate on the fly
            # Use st.spinner so user knows what's happening
            # We don't have st.spinner here easily unless we pass it or assume component context.
            # Since this is cached resource, it runs once.
            
            try:
                # 1. Load
                loader = TextLoader("knowledge.txt", encoding="utf-8")
                documents = loader.load()
                
                # 2. Split
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
                
                # 3. Embed
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
                vector_db = FAISS.from_documents(texts, embeddings)
                
                # 4. Save
                vector_db.save_local("faiss_index")
            except Exception as e:
                # Log error or return None
                 print(f"Failed to build index: {e}")
                 return None
        else:
            return None # No knowledge file

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=API_KEY)
        vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return vector_db
    except Exception as e:
        # st.error(f"Error loading FAISS index: {e}") # Suppress error on login screen
        return None

def get_session_title(messages):
    for m in messages:
        if m["role"] == "user":
            return m["content"][:30] + "..."
    return "New Chat"

# --- AUTHENTICATION LOGIC ---

# Cookie Manager Init
cookie_manager = stx.CookieManager()

# Auto-Login Check (Before Session State Init)
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    
    # Check cookie
    cookie_user = cookie_manager.get("zain_auth_user")
    if cookie_user:
        users = load_users()
        if cookie_user in users:
             st.session_state.authenticated = True
             st.session_state.username = cookie_user
             st.session_state.guest_mode = False

if "username" not in st.session_state:
    st.session_state.username = None
if "guest_mode" not in st.session_state:
    st.session_state.guest_mode = False

def login_page():
    # Custom CSS for "Ebolt" Style Card
    st.markdown("""
        <style>
            /* Hide Streamlit Header/Footer to fix scrolling if possible */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* The Card Container - Targeting the middle column's internal div */
            div[data-testid="column"]:nth-of-type(2) > div {
                background-color: white;
                padding: 40px;
                border-radius: 24px;
                box-shadow: 0 20px 50px rgba(0,0,0,0.1);
                border: 1px solid rgba(0,0,0,0.05);
                margin-top: 5vh; /* Reduced top margin to avoid scroll */
            }
            
            /* Inputs */
            div[data-baseweb="input"] {
                border-radius: 12px;
                background-color: #f3f4f6;
                border: none;
                padding: 8px;
            }
            div[data-baseweb="input"] input {
                background-color: transparent;
                color: #333;
            }
            
            /* Tabs styling - Remove Red Line */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                justify-content: center;
                border-bottom: none !important;
                margin-bottom: 20px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 40px;
                border-radius: 20px;
                padding: 0 20px;
                font-size: 14px;
                font-weight: 500;
                color: #666;
                border: none !important; /* No border overrides */
            }
            .stTabs [aria-selected="true"] {
                background-color: #000 !important;
                color: white !important;
                border-bottom: none !important; /* Explicitly remove red line */
            }
            /* Remove default Streamlit red underline via pseudo-element if existing */
            .stTabs [data-baseweb="tab-highlight"] {
                display: none !important;
            }
            
            /* Primary Button (Get Started) - Force Styles */
            /* Target all primary buttons in the app (only used in login for now) */
            div[data-testid="stAppViewContainer"] button[kind="primary"] {
                background-color: #000000 !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 12px !important;
                height: 50px !important;
                font-weight: 600 !important;
                font-size: 16px !important;
                box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
                transition: all 0.3s ease !important;
            }
            div[data-testid="stAppViewContainer"] button[kind="primary"]:hover {
                background-color: #333333 !important;
                color: #ffffff !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
            }
            
            /* Secondary Button styling (Guest) */
            button[kind="secondary"] {
                background-color: transparent !important;
                color: #555 !important;
                box-shadow: none !important;
                border: 1px solid #eee !important;
            }

            /* Headers */
            h2 {
                font-family: 'Inter', sans-serif;
                font-weight: 700;
                font-size: 26px;
                color: #111;
                text-align: center;
                margin-bottom: 10px;
            }
            p {
                font-family: 'Inter', sans-serif;
                color: #666;
                text-align: center;
                font-size: 15px;
                line-height: 1.5;
            }
            
            /* Icon centering */
            .icon-container {
                display: flex;
                justify-content: center;
                margin-bottom: 20px;
                font-size: 45px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Removed big <br> to fix scroll
    # st.markdown("<br><br>", unsafe_allow_html=True) 
    
    col1, col2, col3 = st.columns([1, 1.2, 1]) # Centered card
    
    with col2:
        # Header Section
        st.markdown("<div class='icon-container'>🧠</div>", unsafe_allow_html=True)
        st.markdown("<h2>Sign in with username</h2>", unsafe_allow_html=True)
        st.markdown("<p>Welcome back to Office Brain. Enter your details to access your assistant.</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Login", "Register", "Guest"])

        with tab1:
            login_user = st.text_input("Username", key="login_user", label_visibility="collapsed", placeholder="Username")
            st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True) # Spacer
            login_pass = st.text_input("Password", type="password", key="login_pass", label_visibility="collapsed", placeholder="Password")
            
            if st.button("Get Started", key="login_btn", type="primary", use_container_width=True):
                users = load_users()
                if login_user in users and users[login_user] == hash_password(login_pass):
                    # Set Cookie (7 days)
                    cookie_manager.set("zain_auth_user", login_user, expires_at=datetime.datetime.now() + datetime.timedelta(days=7))
                    
                    st.session_state.authenticated = True
                    st.session_state.username = login_user
                    st.session_state.guest_mode = False
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab2:
            reg_user = st.text_input("New Username", key="reg_user", label_visibility="collapsed", placeholder="Choose Username")
            st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
            reg_pass = st.text_input("New Password", type="password", key="reg_pass", label_visibility="collapsed", placeholder="Choose Password")
            
            if st.button("Create Account", key="reg_btn", type="primary", use_container_width=True):
                users = load_users()
                if reg_user in users:
                    st.error("Username already exists!")
                elif not reg_user or not reg_pass:
                    st.error("Please fill in all fields")
                else:
                    users[reg_user] = hash_password(reg_pass)
                    save_users(users)
                    st.success("Account created! Switch to Login.")

        with tab3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("No account needed. History is not saved.")
            if st.button("Continue as Guest", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.username = "Guest"
                st.session_state.guest_mode = True
                st.rerun()

# --- MAIN APP FLOW ---

if not st.session_state.authenticated:
    login_page()
else:
    # --- APP INTERFACE (Logged In or Guest) ---
    
    # Init history
    # If User: Load from file
    # If Guest: Use empty dict (in-memory only)
    if "full_history" not in st.session_state:
        if st.session_state.guest_mode:
            st.session_state.full_history = {} # Start fresh for guest
        else:
            st.session_state.full_history = load_history(st.session_state.username)

    # Init current session
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- SIDEBAR ---
    # --- SIDEBAR ---
    # --- SIDEBAR ---
    # --- SIDEBAR ---
    with st.sidebar:
        # Load FontAwesome & Custom CSS
        st.markdown("""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
            <style>
                /* SIDEBAR LAYOUT */
                section[data-testid="stSidebar"] > div:nth-child(2) {
                    display: flex;
                    flex-direction: column;
                    height: 100vh;
                }
                
                div[data-testid="stSidebarHeader"] {display: none;}
                
                /* 1. HEADER */
                .sidebar-header {
                    margin-bottom: 20px;
                    margin-top: 10px;
                    padding-left: 5px;
                }
                .logo-text {
                    font-size: 22px;
                    font-weight: 700;
                    color: #1a1a1a;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .logo-icon { font-size: 26px; }
                .welcome-text {
                    font-size: 13px;
                    color: #666;
                    margin-left: 4px;
                    margin-top: 5px;
                }
                
                /* 2. BUTTONS (TRANSPARENT & FLAT) */
                div[data-testid="stSidebar"] button {
                    background-color: transparent !important;
                    border: none !important;
                    box-shadow: none !important;
                    color: #444 !important;
                    text-align: left !important;
                    display: flex !important;
                    justify-content: flex-start !important;
                    padding-left: 10px !important;
                    transition: background-color 0.2s, color 0.2s !important;
                }
                
                div[data-testid="stSidebar"] button:hover {
                    background-color: #f0f2f6 !important;
                    color: #000 !important;
                    border: none !important;
                    box-shadow: none !important;
                }
                
                /* Fix text internal structure */
                div[data-testid="stSidebar"] button p {
                    font-weight: 500 !important;
                }

                /* 3. DELETE BUTTON (ALIGNMENT FIX) */
                div[data-testid="column"]:nth-of-type(2) button {
                    color: #ff4b4b !important;
                    padding: 0 !important;
                    margin-top: 10px !important; /* Visual Push Down */
                    justify-content: center !important;
                }
                
                /* 4. SCROLLABLE HISTORY WRAPPER */
                .history-scroll-area {
                    flex: 1;
                    overflow-y: auto;
                    min-height: 100px;
                    margin-top: 10px;
                    scrollbar-width: thin;
                }
                
                /* 5. FOOTER (LOGOUT) */
                .logout-footer {
                    margin-top: auto;
                    padding-top: 20px;
                    padding-bottom: 20px;
                    border-top: 1px solid #eaeaea;
                    background-color: transparent;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # --- HEADER ---
        st.markdown(f"""
        <div class="sidebar-header">
            <div class="logo-text">
                <span class="logo-icon">🧠</span> Office Brain
            </div>
            <div class="welcome-text">
                Welcome, <b>{st.session_state.username if st.session_state.username else 'Guest'}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- NEW CHAT ---
        if st.button("✏️ New Chat", key="new_chat_btn", use_container_width=True):
             st.session_state.current_session_id = str(uuid.uuid4())
             st.session_state.messages = []
             st.rerun()

        # --- HISTORY AREA ---
        st.markdown('<div class="history-scroll-area">', unsafe_allow_html=True)
        
        sessions_list = []
        for sid, data in st.session_state.full_history.items():
            sessions_list.append({"id": sid, "title": data.get("title", "Untitled"), "timestamp": data.get("timestamp", "")})

        if not sessions_list and st.session_state.guest_mode:
             st.markdown("<p style='color:#999; font-size:13px; padding-left:10px;'>Guest chats are temporary.</p>", unsafe_allow_html=True)

        for session in reversed(sessions_list):
            # Adjusted column ratio for better delete icon spacing
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                display_title = session['title']
                if len(display_title) > 20:
                    display_title = display_title[:18] + ".."
                
                # Simple label
                label_title = f"💬 {display_title}"
                if st.button(label_title, key=session["id"], use_container_width=True):
                    st.session_state.current_session_id = session["id"]
                    st.session_state.messages = st.session_state.full_history[session["id"]]["messages"]
                    st.rerun()
            with col2:
                # Using a smaller key/help to render delete
                if st.button("🗑️", key=f"del_{session['id']}", help="Delete chat"):
                    if session["id"] in st.session_state.full_history:
                        del st.session_state.full_history[session["id"]]
                        if not st.session_state.guest_mode:
                            save_history(st.session_state.full_history, st.session_state.username)
                        if st.session_state.current_session_id == session["id"]:
                            st.session_state.current_session_id = str(uuid.uuid4())
                            st.session_state.messages = []
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True) # End scroll area
        
        # --- LOGOUT FOOTER ---
        st.markdown('<div class="logout-footer">', unsafe_allow_html=True)
        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            cookie_manager.delete("zain_auth_user")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- MAIN CHAT ---
    
    # Load Knowledge Base (FAISS)
    vector_db = load_faiss_index()

    # Determine Icons
    # Using flaticon/icons8 safe URLs
    bot_icon = "https://cdn-icons-png.flaticon.com/512/4712/4712035.png" # Brain/Bot
    user_icon = "https://cdn-icons-png.flaticon.com/512/9131/9131529.png" # User

    st.title("💬 Ask Me Anything about office")

    base_system_prompt = """
    You are a helpful and friendly Office Assistant AI.
    Answer questions based ONLY on the provided context below. 
    Keep answers short and professional.
    """

    # Display History
    for message in st.session_state.messages:
        avatar = bot_icon if message["role"] == "assistant" else user_icon
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=user_icon):
            st.markdown(prompt)

        # Generate Response
        try:
            # 1. Retrieve Context
            context = ""
            if vector_db:
                 try:
                    docs = vector_db.similarity_search(prompt, k=4)
                    context = "\n\n".join([d.page_content for d in docs])
                 except:
                    context = "" # Fallback if FAISS fails
            
            # 2. Prepare System Prompt
            full_system_prompt = f"{base_system_prompt}\n\nCONTEXT:\n{context}"

            model = genai.GenerativeModel(
                model_name="gemini-flash-latest",
                generation_config=generation_config,
                system_instruction=full_system_prompt
            )
            
            # Convert history for Gemini
            gemini_history = []
            for m in st.session_state.messages[:-1]:
                 role = "user" if m["role"] == "user" else "model"
                 gemini_history.append({"role": role, "parts": [m["content"]]})
            
            chat_session = model.start_chat(history=gemini_history)
            
            with st.chat_message("assistant", avatar=bot_icon):
                with st.spinner("Thinking..."):
                    response = chat_session.send_message(prompt)
                    response_text = response.text
                    st.markdown(response_text)
                
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # --- SAVE TO HISTORY ---
            session_title = get_session_title(st.session_state.messages)
            st.session_state.full_history[st.session_state.current_session_id] = {
                "title": session_title,
                "messages": st.session_state.messages,
                "timestamp": str(datetime.datetime.now())
            }
            
            # Save ONLY if not guest
            if not st.session_state.guest_mode:
                save_history(st.session_state.full_history, st.session_state.username)
            
        except Exception as e:
            if "429" in str(e):
                 st.error("Too many requests! Please wait a moment.")
            else:
                 st.error(f"An error occurred: {e}")
