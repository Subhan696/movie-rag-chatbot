# 📦 Import necessary libraries
import os
import pandas as pd
import google.generativeai as genai  # ✅ Gemini SDK
from dotenv import load_dotenv
import gradio as gr
from typing import List, Dict, Any
import requests  # For TMDB API

# Vector search client
from astrapy import DataAPIClient

# 🔷 Load env and Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 📁 Paths
EXCEL_FILE = "user_info.xlsx"

# 🔷 AstraDB connection
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT", "")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
if not ASTRA_DB_API_ENDPOINT or not ASTRA_DB_APPLICATION_TOKEN:
    raise RuntimeError("Missing ASTRA_DB_* env vars. Check .env")

client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
database = client.get_database(ASTRA_DB_API_ENDPOINT)
collection = database.get_collection("movies")

# Embedding model info
EMBEDDING_MODEL = "text-embedding-004"
VECTOR_DIMENSION = 768

def embed_text(text: str) -> List[float]:
    """Get Gemini embedding vector for user query or document text."""
    res = genai.embed_content(model=EMBEDDING_MODEL, content=(text or ""))
    emb = res.get("embedding") if isinstance(res, dict) else getattr(res, "embedding", None)
    if isinstance(emb, dict) and "values" in emb:
        return emb["values"]
    return emb

def retrieve_relevant_movies(user_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Vector similarity search in AstraDB on `embedding` field."""
    query_embedding = embed_text(user_query)
    if not isinstance(query_embedding, list):
        return []
    # Vector sort uses the special $vector key
    results = collection.find(
        {},
        sort={"$vector": query_embedding},
        limit=top_k,
        include_similarity=True,
    )
    # astrapy returns a cursor-like iterable of dicts
    return list(results)

# TMDB API setup
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_API_URL = "https://api.themoviedb.org/3"

def fetch_movie_from_tmdb(query: str) -> dict:
    """Fetch movie details from TMDB by search query."""
    if not TMDB_API_KEY:
        return {}
    search_url = f"{TMDB_API_URL}/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": query}
    resp = requests.get(search_url, params=params)
    if resp.status_code != 200:
        return {}
    results = resp.json().get("results", [])
    if not results:
        return {}
    movie = results[0]  # Take the top result
    # Fetch more details
    details_url = f"{TMDB_API_URL}/movie/{movie['id']}"
    details_params = {"api_key": TMDB_API_KEY, "append_to_response": "credits"}
    details_resp = requests.get(details_url, params=details_params)
    if details_resp.status_code != 200:
        return movie
    details = details_resp.json()
    # Extract director and cast
    director = ""
    cast = []
    for member in details.get("credits", {}).get("crew", []):
        if member.get("job") == "Director":
            director = member.get("name")
            break
    for actor in details.get("credits", {}).get("cast", [])[:5]:
        cast.append(actor.get("name"))
    # Compose movie info
    return {
        "title": details.get("title", movie.get("title")),
        "year": details.get("release_date", "?")[:4],
        "description": details.get("overview", movie.get("overview", "")),
        "director": director,
        "cast": cast,
        "genre": ", ".join([g["name"] for g in details.get("genres", [])]),
        "box_office": details.get("revenue", "Unknown"),
        "imdb_rating": details.get("vote_average", 0),
        "runtime_min": details.get("runtime", 0),
        "streaming": "Unknown"  # TMDB does not provide streaming info directly
    }

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
        f"Use only the retrieved movie data unless asked general knowledge. "
        f"Refer to chat history to keep continuity in your responses."
    )

    recent_history = session["chat_history"][-5:]
    history_text = ""
    for turn in recent_history:
        role = turn["role"].capitalize()
        history_text += f"{role}: {turn['content']}\n"

    # Retrieve relevant context via vector search
    retrieved = retrieve_relevant_movies(user_query, top_k=5)
    context_blocks = []
    for m in retrieved:
        context_blocks.append(
            f"{m.get('title','N/A')} ({m.get('year','?')}): {m.get('description','')}. "
            f"Director: {m.get('director','')}. Cast: {', '.join(m.get('cast', []) if isinstance(m.get('cast'), list) else [])}. "
            f"Genre: {m.get('genre','')}. Box Office: {m.get('box_office','Unknown')}. "
            f"IMDb: {m.get('imdb_rating',0)}. Available on: {m.get('streaming','Unknown')}. "
            f"Length: {m.get('runtime_min',0)} minutes."
        )
    # If no relevant movies found, try TMDB
    if not context_blocks:
        tmdb_movie = fetch_movie_from_tmdb(user_query)
        if tmdb_movie:
            context_blocks.append(
                f"{tmdb_movie.get('title','N/A')} ({tmdb_movie.get('year','?')}): {tmdb_movie.get('description','')}. "
                f"Director: {tmdb_movie.get('director','')}. Cast: {', '.join(tmdb_movie.get('cast', []))}. "
                f"Genre: {tmdb_movie.get('genre','')}. Box Office: {tmdb_movie.get('box_office','Unknown')}. "
                f"IMDb: {tmdb_movie.get('imdb_rating',0)}. Available on: {tmdb_movie.get('streaming','Unknown')}. "
                f"Length: {tmdb_movie.get('runtime_min',0)} minutes."
            )
    movie_context = "\n".join(context_blocks) if context_blocks else "(No relevant movies found in DB or TMDB)"

    prompt = (
        f"{system_message}\n\n"
        f"### CHAT HISTORY:\n{history_text}\n"
        f"### RETRIEVED MOVIES:\n{movie_context}\n"
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