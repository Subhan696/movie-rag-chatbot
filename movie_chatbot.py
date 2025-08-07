# 📦 Import necessary libraries
import os
import pandas as pd
import google.generativeai as genai  # ✅ Gemini SDK
from dotenv import load_dotenv
import gradio as gr

# 🔷 Load Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 📁 Paths
MOVIE_PATH = r"H:\\Subhan\\Hollywood_Top_Movies.xlsx"
EXCEL_FILE = "user_info.xlsx"

# 🔷 Load Movie Data
df = pd.read_excel(MOVIE_PATH)

# Prepare the movie knowledge text
movie_knowledge = ""
for _, row in df.iterrows():
    movie_knowledge += (
        f"{row['Movie Name']} ({row['Year']}): {row['Description']}. "
        f"Director: {row['Director']}. Cast: {row['Cast']}. Genre: {row['Genre']}. "
        f"Box Office: {row['Box Office']}. IMDb: {row['IMDb Rating']}. "
        f"Available on: {row['Streaming']}. Length: {row['Runtime (min)']} minutes.\n"
    )

# 🔷 Session state
def init_session():
    return {
        "name": None, "phone": None, "email": None, "location": None,
        "collected": False,
        "chat_history": [],
        "first_prompt": True
    }

# 🔷 Save user info
def save_user_info(session):
    user_info = {
        "Name": session["name"],
        "Phone": session["phone"],
        "Email": session["email"],
        "Location": session["location"]
    }
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
        df = pd.concat([df, pd.DataFrame([user_info])], ignore_index=True)
    else:
        df = pd.DataFrame([user_info])
    df.to_excel(EXCEL_FILE, index=False)

# 🔷 Gemini query using full chat history + personalization
def query_gpt(user_query, session):
    user_name = session.get("name", "User")
    user_location = session.get("location", "their location")

    system_message = (
        f"You are a friendly and helpful assistant answering questions about Hollywood movies. "
        f"The user you're helping is named {user_name} from {user_location}. "
        f"Make your responses conversational and, when natural, refer to them by name. "
        f"Use only the movie dataset unless asked general knowledge. "
        f"Refer to chat history to keep continuity in your responses."
    )

    recent_history = session["chat_history"][-5:]
    history_text = ""
    for turn in recent_history:
        role = turn["role"].capitalize()
        history_text += f"{role}: {turn['content']}\n"

    prompt = (
        f"{system_message}\n\n"
        f"### CHAT HISTORY:\n{history_text}\n"
        f"### MOVIE DATA:\n{movie_knowledge}\n"
        f"### USER QUESTION:\n{user_query}"
    )

    model = genai.GenerativeModel(model_name="gemini-2.0-flash-001")
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Sorry {user_name}, I encountered an error while fetching the response. Please try again later."

# 🔷 Handle input from user
def handle_input(user_input, session):
    if session["first_prompt"]:
        session["first_prompt"] = False
        return "Hello! \U0001F44B\nMay I have your name, please?\n", session

    if not session["collected"]:
        if not session["name"]:
            session["name"] = user_input.strip()
            return "Thank you, may I have your phone number?", session
        elif not session["phone"]:
            session["phone"] = user_input.strip()
            return "Great. May I have your email address?", session
        elif not session["email"]:
            session["email"] = user_input.strip()
            return "Thank you. Lastly, may I know your location?", session
        elif not session["location"]:
            session["location"] = user_input.strip()
            session["collected"] = True
            save_user_info(session)
            return (
                f"Awesome, {session['name']} from {session['location']}! \U0001F389\n"
                f"You can now ask me anything about Hollywood movies — cast, director, box office, or where to stream them. "
                f"Let's get started! \U0001F37F"
            ), session

    session["chat_history"].append({"role": "user", "content": user_input})
    movie_response = query_gpt(user_input, session)
    session["chat_history"].append({"role": "assistant", "content": movie_response})

    return movie_response, session

# 🔷 Gradio callbacks
def on_submit(user_message, chat_history, state):
    response, state = handle_input(user_message, state)
    chat_history.append([user_message, response])
    return "", chat_history, state

def on_clear():
    return [], "", init_session()

# 🔷 Gradio app
with gr.Blocks() as demo:
    gr.Markdown("## \U0001F3AC Hollywood Movie Chatbot + User Info")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Message", placeholder="Hi!")
    clear = gr.Button("Clear Chat")
    state = gr.State(init_session())

    msg.submit(on_submit, [msg, chatbot, state], [msg, chatbot, state])
    clear.click(on_clear, [], [chatbot, msg, state])

demo.launch()
