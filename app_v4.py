import os
import sqlite3
import bcrypt
import requests
from bs4 import BeautifulSoup
import spacy
from collections import Counter
from textblob import TextBlob
import gradio as gr
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

load_dotenv()

# Initialize Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up SQLite database connection (persistent storage)
DB_PATH = "users.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Create users table if it doesn't exist
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password BLOB
    )
    """
)

# Create history table if it doesn't exist
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        url TEXT,
        summary TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
)
conn.commit()

# -------------------------
# NLP & Analysis Functions
# -------------------------

def semantic_analyzer(url):
    """Improved semantic analysis with content filtering."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

    try:
        soup = BeautifulSoup(response.content, "html.parser")
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        text = ' '.join(soup.stripped_strings)
    except Exception as e:
        return {"error": f"Content parsing error: {str(e)}"}

    doc = nlp(text[:100000])  # Limit processing to first 100k characters

    relevant_entities = [
        (ent.text, ent.label_)
        for ent in doc.ents
        if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT']
    ]

    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop
    ][:20]

    return {
        "entities": relevant_entities[:15],
        "keywords": list(set(keywords))[:10],
        "sentiment": TextBlob(text).sentiment.polarity,
        "word_freq": Counter([token.text.lower() for token in doc if token.is_alpha])
    }

def generate_summary(analysis, summary_length):
    """Generate summary using Gemini API with dynamic headings."""
    if "error" in analysis:
        return f"⚠️ Error: {analysis['error']}"

    # Instruct the AI to generate dynamic headings for each paragraph
    system_prompt = (
        "You are a senior news analyst. Generate a concise summary of the analysis results in three paragraphs. "
        "For each paragraph, include a heading that captures the main topic of that paragraph. "
        f"Keep the entire summary under {int(summary_length)} words and use a journalistic tone."
    )

    user_prompt = (
        f"**Analysis Results:**\n"
        f"- **Overall Sentiment:** {analysis['sentiment']:.2f}\n"
        f"- **Key Entities:** {', '.join([f'{e[0]} ({e[1]})' for e in analysis['entities']])}\n"
        f"- **Top Keywords:** {', '.join(analysis['keywords'])}\n"
        f"- **Frequent Terms:** {', '.join([f'{k} ({v})' for k, v in analysis['word_freq'].most_common(5)])}"
    )

    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Generation Error: {str(e)}"

def process_url(url, summary_length):
    """Processing pipeline: Semantic analysis and summary generation."""
    analysis = semantic_analyzer(url)
    return generate_summary(analysis, summary_length)

def process_and_save(url, summary_length, username):
    """Process URL, generate summary, and save the analysis to history for the user."""
    summary = process_url(url, summary_length)
    # Save history only if a valid summary is generated (do not store errors)
    if not summary.startswith("⚠️"):
        cursor.execute(
            "INSERT INTO history (username, url, summary) VALUES (?, ?, ?)",
            (username, url, summary)
        )
        conn.commit()
    return summary

def get_user_history(username):
    """Retrieve analysis history for the given username."""
    cursor.execute(
        "SELECT url, summary, timestamp FROM history WHERE username = ? ORDER BY timestamp DESC",
        (username,)
    )
    rows = cursor.fetchall()
    history = [list(row) for row in rows]
    return history

def get_user_history_html(username):
    """Return user history as an HTML table with scrollable summaries and newlines preserved."""
    rows = get_user_history(username)  # Each row: [url, summary, timestamp]
    html = '<table style="width:100%; border-collapse: collapse;">'
    html += (
        '<thead><tr style="border-bottom: 1px solid #ddd;">'
        '<th style="padding: 8px; text-align: left;">URL</th>'
        '<th style="padding: 8px; text-align: left;">Summary</th>'
        '<th style="padding: 8px; text-align: left;">Timestamp</th>'
        '</tr></thead><tbody>'
    )
    for row in rows:
        url, summary, timestamp = row
        # Wrap summary in a scrollable container with newlines preserved
        summary_div = f'<div style="max-height: 100px; overflow-y: auto; white-space: pre-wrap;">{summary}</div>'
        html += (
            f'<tr style="border-bottom: 1px solid #ddd;">'
            f'<td style="padding: 8px; vertical-align: top;">{url}</td>'
            f'<td style="padding: 8px; vertical-align: top;">{summary_div}</td>'
            f'<td style="padding: 8px; vertical-align: top;">{timestamp}</td>'
            '</tr>'
        )
    html += '</tbody></table>'
    return html

# -------------------------
# User Management Functions
# -------------------------

def signup(username, password, confirm_password):
    """Register a new user with encrypted password."""
    if not username or not password:
        return "Username and password are required."
    if password != confirm_password:
        return "Passwords do not match."
    
    cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        return "Username already exists."

    salt = bcrypt.gensalt()
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), salt)
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
    conn.commit()
    return "Sign up successful! Please log in."

def login(username, password):
    """Authenticate a user using encrypted password."""
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    if result and bcrypt.checkpw(password.encode('utf-8'), result[0]):
        return username, f"Welcome, {username}!"
    return "", "Invalid username or password."

# -------------------------
# Custom CSS
# -------------------------
custom_css = """
/* Center the main container and limit its width for a modern look */
.gradio-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    max-width: 800px;
    margin: auto;
    padding: 2em;
}

/* General font and background styling */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f4f7f6;
    color: #333;
}

/* Style for Gradio buttons */
.gr-button {
    background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
    color: #333;
    border: none;
    transition: background 0.3s ease;
    padding: 0.8em 1.2em;
    font-size: 1em;
}
.gr-button:hover {
    background: linear-gradient(135deg, #89b4f7 0%, #a3d0fb 100%);
}

/* Smaller login button */
#login_button {
    padding: 0.4em 0.8em;
    font-size: 0.9em;
}

/* Additional margin for inputs and elements */
.gradio-container .input, .gradio-container .gr-button {
    margin-bottom: 1em;
}

/* Custom styling for the history table */
.history-table {
    width: 100%;
    border-collapse: collapse;
}
.history-table th, .history-table td {
    padding: 8px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}
"""

# -------------------------
# Gradio Interface
# -------------------------
with gr.Blocks(css=custom_css, title="News Semantic Analyzer") as demo:
    # State to store the current logged-in user (empty string if not logged in)
    current_user = gr.State("")

    # -------------------------
    # Login / Signup Panel
    # -------------------------
    with gr.Column(visible=True, elem_id="login_container") as login_container:
        gr.Markdown("# Welcome to News Semantic Analyzer")
        gr.Markdown("Please **Log In** or **Sign Up** to continue.")
        
        with gr.Tabs():
            # Log In Tab
            with gr.TabItem("Log In"):
                login_username = gr.Textbox(label="Username", placeholder="Enter your username")
                login_password = gr.Textbox(label="Password", placeholder="Enter your password", type="password")
                login_button = gr.Button("Log In", elem_id="login_button")
                login_message = gr.Markdown("")
            
            # Sign Up Tab
            with gr.TabItem("Sign Up"):
                signup_username = gr.Textbox(label="Username", placeholder="Choose a username")
                signup_password = gr.Textbox(label="Password", placeholder="Enter a password", type="password")
                signup_confirm = gr.Textbox(label="Confirm Password", placeholder="Re-enter your password", type="password")
                signup_button = gr.Button("Sign Up")
                signup_message = gr.Markdown("")

    # -------------------------
    # Main Application Panel (hidden until login)
    # -------------------------
    with gr.Column(visible=False, elem_id="main_container") as main_container:
        with gr.Row():
            with gr.Column(scale=8):
                gr.Markdown("## News Analysis")
            with gr.Column(scale=2):
                logout_button = gr.Button("Logout", variant="stop")
        with gr.Tabs():
            # Analysis Tab
            with gr.TabItem("Analysis"):
                with gr.Row():
                    with gr.Column(scale=2):
                        url_input = gr.Textbox(
                            label="News URL",
                            placeholder="Enter valid news website URL...",
                            show_label=True
                        )
                        summary_length = gr.Slider(
                            minimum=50,
                            maximum=500,
                            step=10,
                            value=200,
                            label="Summary Length (in words)"
                        )
                        analyze_button = gr.Button("Analyze", variant="primary")
                    with gr.Column(scale=6):
                        summary_output = gr.Textbox(
                            label="Analysis Summary",
                            placeholder="The generated summary will appear here...",
                            lines=10,
                            show_copy_button=True
                        )
            # History Tab
            with gr.TabItem("History"):
                history_button = gr.Button("Refresh History")
                # Use an HTML component to display a table with scrollable summary sections
                history_output = gr.HTML()

        gr.Markdown(
            """
            <div style="text-align: center; margin-top: 1em;">
                <small>Powered by spaCy, TextBlob, Gemini AI, and Gradio</small>
            </div>
            """
        )

    # -------------------------
    # Event Handlers
    # -------------------------
    def perform_signup(username, password, confirm):
        return signup(username, password, confirm)
    
    def perform_login(username, password):
        user, msg = login(username, password)
        if user:
            # Successful login: hide the login panel and show the main application panel.
            return user, gr.update(visible=False), gr.update(visible=True), msg
        else:
            return "", gr.update(visible=True), gr.update(visible=False), msg

    def perform_logout(_):
        # On logout: clear current user and switch back to the login panel.
        return "", gr.update(visible=True), gr.update(visible=False), "Logged out."

    def refresh_history(username):
        # Retrieve history as an HTML table to be displayed
        return get_user_history_html(username)

    # Bind the signup, login, analyze, history refresh, and logout actions
    signup_button.click(
        fn=perform_signup,
        inputs=[signup_username, signup_password, signup_confirm],
        outputs=signup_message
    )

    login_button.click(
        fn=perform_login,
        inputs=[login_username, login_password],
        outputs=[current_user, login_container, main_container, login_message]
    )

    analyze_button.click(
        fn=process_and_save,
        inputs=[url_input, summary_length, current_user],
        outputs=summary_output
    )

    history_button.click(
        fn=refresh_history,
        inputs=current_user,
        outputs=history_output
    )

    logout_button.click(
        fn=perform_logout,
        inputs=current_user,
        outputs=[current_user, login_container, main_container, login_message]
    )

if __name__ == "__main__":
    demo.launch(server_port=int(os.getenv("PORT", 7860)), share=True)