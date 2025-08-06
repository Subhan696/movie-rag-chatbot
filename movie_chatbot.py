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
MOVIE_PATH = r"H:\Subhan\Hollywood_Top_Movies.xlsx"
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

# 🔷 Sample prompts
SAMPLE_PROMPTS = [
    "Who directed Titanic?",
    "Which movie has the highest box office earnings?",
    "Name a movie starring Leonardo DiCaprio.",
    "Which movies were released in 1994?",
    "What is the IMDb rating of The Godfather?",
    "Which movies are available on Netflix?",
    "List movies in the Action genre.",
    "Who acted in Inception?",
    "What is the runtime of Avatar?",
    "Which movie won the most Oscars?",
    "Tell me about the movie The Dark Knight.",
    "Who is the director of Jurassic Park?",
    "What are the movies in the Comedy genre?",
    "Which movies feature Tom Hanks?",
    "Give me details about the movie Forrest Gump.",
    "Which movie has the longest runtime?",
    "List movies directed by Steven Spielberg.",
    "What is the description of the movie Interstellar?",
    "Name a movie released after 2010 with high IMDb rating.",
    "Which movies are available on Amazon Prime?"
]

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

# 🔷 Gemini query
def query_gpt(user_query):
    """
    Sends user’s question to Gemini AI with the movie dataset and returns the response.
    """
    system_message = (
        "You are a helpful assistant with access to a Hollywood movie dataset. "
        "First, answer if the data is available in the dataset provided. "
        "If the answer is not in the dataset, check from outside the dataset using the Gemini model."
    )

    prompt = f"{system_message}\n\nHere is the movie data:\n{movie_knowledge}\n\nQuestion: {user_query}"

    model = genai.GenerativeModel(model_name="gemini-2.0-flash-001")  # ✅ Correct model path
    response = model.generate_content(prompt)
    return response.text.strip()

# 🔷 Handle input
def handle_input(user_input, session):
    if session["first_prompt"]:
        session["first_prompt"] = False
        welcome_message = (
            "Hello! 👋\n"
            "May I have your name, please?\n\n"
        )
        return welcome_message, session

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
            return (f"Thank you, {session['name']} from {session['location']}.\n"
                    "You can now ask me anything about the Hollywood movies in our dataset!"), session

    session["chat_history"].append({"role": "user", "content": user_input})
    movie_response = query_gpt(user_input)
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
    gr.Markdown("## 🎬 Hollywood Movie Chatbot + User Info")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Message", placeholder="Hi!")
    clear = gr.Button("Clear Chat")
    state = gr.State(init_session())

    msg.submit(on_submit, [msg, chatbot, state], [msg, chatbot, state])
    clear.click(on_clear, [], [chatbot, msg, state])

demo.launch()
