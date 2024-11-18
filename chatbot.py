import streamlit as st
import torch
from transformers import pipeline
import os
import json
from huggingface_hub import login

# Initialize the user database file
USER_DB_FILE = "users.txt"
CHAT_HISTORY_DIR = "chat_histories"

api_key = os.getenv("HF_API_KEY")
login(api_key)

    
# Function to load user credentials from a file
def load_users():
    if not os.path.exists(USER_DB_FILE):
        return {}
    with open(USER_DB_FILE, "r") as file:
        users = {}
        for line in file:
            username, password = line.strip().split(":")
            users[username] = password
        return users

# Function to save a new user to the file
def save_user(username, password):
    with open(USER_DB_FILE, "a") as file:
        file.write(f"{username}:{password}\n")

# Function to load a user's chat history from a file
def load_chat_history(username):
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{username}_history.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return []  # Return an empty list if no history exists

# Function to save a user's chat history to a file
def save_chat_history(username, chat_history):
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{username}_history.json")
    with open(file_path, "w") as file:
        json.dump(chat_history, file)

# Initialize session states
if "users" not in st.session_state:
    st.session_state.users = load_users()  # Store user credentials: {username: password}
if "current_user" not in st.session_state:
    st.session_state.current_user = None  # Track logged-in user
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store chat history for the logged-in user

# Set the device for model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model pipeline
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device=device,
)

# Function to handle login
def login_user(username, password):
    users = load_users()  
    if username in users and users[username] == password:
        return True
    return False

# Function to handle signup
def signup_user(username, password):
    users = load_users()  
    if username not in users:
        save_user(username, password)  # Save new user to the file
        return True
    return False

# Function to handle chat
def chat_with_model(user_input, chat_history):
    
    # Add user message to chat history
    chat_history.append({"role": "user", "content": user_input})
    
    # Prepare prompt for model by combining all messages in chat history
    prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]) + "\nAssistant:"
    
    # Generate a response
    outputs = pipe(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # Extract the assistant's response from generated text
    assistant_response = outputs[0]["generated_text"].split("Assistant:")[-1].strip()
    
    # Add the assistant's response to the chat history
    chat_history.append({"role": "assistant", "content": assistant_response})
    return assistant_response

st.markdown(
        """
        <style>
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 0 !important;
            border: none !important;
            background: none !important;
            margin; 0 !important;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            text-align: right;
            margin: 5px 0;
        }
        .chat-message.assistant {
            text-align: left;
            margin: 5px 0;
        }
        .chat-message span {
            display: inline-block;
            padding: 10px;
            border-radius: 10px;
            max-width: 80%;
        }
        .chat-message.user span {
            background-color: #d1e7dd;
            color: #0f5132;
        }
        .chat-message.assistant span {
            background-color: #f8d7da;
            color: #842029;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Display chat history
def render_chat_history(username):
    chat_history = load_chat_history(username)

    # Return immediately if no history exists
    if not chat_history:
        return
    
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    # Display each message in the chat history
    for chat in chat_history:
        if chat["role"] == "user":
            st.markdown(
                f"<div class='chat-message user'><span>{chat['content']}</span></div>",
                unsafe_allow_html=True,
            )
        elif chat["role"] == "assistant":
            st.markdown(
                f"<div class='chat-message assistant'><span>{chat['content']}</span></div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

# App layout
st.title("Chat Assistant")

if st.session_state.current_user is None:
    # Login/Signup form
    st.subheader("Login or Signup")
    username = st.text_input("Username", key="username")
    password = st.text_input("Password", type="password", key="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login"):
            if username and password:
                if login_user(username, password):  # Check if login is successful
                    st.session_state.current_user = username
                    st.session_state.chat_history = load_chat_history(username) or [
                        {"role": "system", "content": "You are a friendly assistant who answers the user's questions in a helpful way!"}
                    ]  # Reset chat history for new session
                    st.success("Login successful!")
                else:
                    st.error("Invalid username or password. Please try again.")
            else:
                st.error("Please enter both username and password.")

    with col2:
        if st.button("Signup"):
            if username and password:
                if signup_user(username, password):  # Check if signup is successful
                    st.session_state.users[username] = password
                    st.success("Signup successful! You can now log in.")
                else:
                    st.error("Username already exists. Please choose a different one.")
            else:
                st.error("Please enter both username and password.")
else:
    # Chat interface
    st.subheader(f"Welcome, {st.session_state.current_user}!")
    
    if "chat_history_rendered" not in st.session_state:
     # Render chat history dynamically
        if st.session_state.chat_history:
            render_chat_history(st.session_state.current_user)  
        st.session_state.chat_history_rendered = True
    
    # User input with placeholder and pressing Enter for submission
    user_input = st.text_input("", placeholder="Ask something...", label_visibility="collapsed")
    if user_input:
            st.markdown(
                f"<div class='chat-message user'><span>{user_input}</span></div>",
                    unsafe_allow_html=True,
                )
            assistant_response = chat_with_model(user_input, st.session_state.chat_history)
            st.markdown(
                f"<div class='chat-message assistant'><span>{assistant_response}</span></div>",
                unsafe_allow_html=True,
            )
            save_chat_history(st.session_state.current_user, st.session_state.chat_history)        
    
    # Logout button
    if st.button("Logout"):
        save_chat_history(st.session_state.current_user, st.session_state.chat_history)  # Save chat history before logging out
        st.session_state.current_user = None
        st.session_state.chat_history = []
        if "chat_history_rendered" in st.session_state:
            del st.session_state.chat_history_rendered
