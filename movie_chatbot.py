# 📦 Import necessary libraries
import os
import pandas as pd
import google.generativeai as genai  # ✅ Gemini SDK
from dotenv import load_dotenv
import gradio as gr
from typing import List, Dict, Any
import requests  # For TMDB API
import csv

# Vector search client
from astrapy import DataAPIClient

# 🔷 Load env and Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 📁 Paths
EXCEL_FILE = "user_info.xlsx"
FEEDBACK_FILE = "chatbot_feedback.csv"

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
        "first_prompt": True,
        "last_movie": None,
        "last_movie_context": None,
        "preferences": {"genres": set(), "actors": set(), "directors": set()}
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

# 🔷 Save feedback
def save_feedback(user, message, response, rating):
    """Save feedback to a CSV file."""
    row = {
        "User": user,
        "Message": message,
        "Response": response,
        "Rating": rating
    }
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

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

def update_preferences(session, movie_info):
    """Update user preferences in session based on movie info."""
    if not movie_info:
        return
    prefs = session.setdefault("preferences", {"genres": set(), "actors": set(), "directors": set()})
    # Add genres
    genres = movie_info.get("genre", "")
    if genres:
        for g in genres.split(","):
            prefs["genres"].add(g.strip())
    # Add actors
    for actor in movie_info.get("cast", []):
        prefs["actors"].add(actor)
    # Add director
    director = movie_info.get("director", "")
    if director:
        prefs["directors"].add(director)

def recommend_movies(session, top_k=3):
    """Recommend movies based on user preferences using AstraDB or TMDB."""
    prefs = session.get("preferences", {})
    # Try AstraDB vector search first
    query_parts = []
    if prefs.get("genres"):
        query_parts.append(f"genre: {', '.join(list(prefs['genres']))}")
    if prefs.get("actors"):
        query_parts.append(f"cast: {', '.join(list(prefs['actors']))}")
    if prefs.get("directors"):
        query_parts.append(f"director: {', '.join(list(prefs['directors']))}")
    query = ", ".join(query_parts) if query_parts else "popular movies"
    results = retrieve_relevant_movies(query, top_k=top_k)
    if not results:
        # Fallback: TMDB discover API
        if not TMDB_API_KEY:
            return []
        discover_url = f"{TMDB_API_URL}/discover/movie"
        params = {"api_key": TMDB_API_KEY, "sort_by": "popularity.desc", "page": 1}
        if prefs.get("genres"):
            # TMDB genre IDs would be needed for more accuracy
            pass
        resp = requests.get(discover_url, params=params)
        if resp.status_code != 200:
            return []
        movies = resp.json().get("results", [])[:top_k]
        return [
            {
                "title": m.get("title"),
                "year": m.get("release_date", "?")[:4],
                "description": m.get("overview", "")
            } for m in movies
        ]
    return results

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

    # --- Context Awareness: Track last discussed movie ---
    session.setdefault("last_movie", None)
    session.setdefault("last_movie_context", None)

    # Add user input to chat history
    session["chat_history"].append({"role": "user", "content": user_input})

    # Try to resolve follow-up/ambiguous queries
    import re
    followup_patterns = [
        r"who (directed|is the director)",
        r"who (starred|acted|is in it)",
        r"what (is|was) the box office",
        r"what (is|was) the genre",
        r"how long (is|was) it",
        r"where can i watch|stream (it|this|that)"
    ]
    is_followup = any(re.search(pat, user_input, re.I) for pat in followup_patterns)
    if is_followup and session["last_movie_context"]:
        # Use last movie context for the answer
        movie_response = session["last_movie_context"]
    else:
        # Normal flow: get response and update last_movie if possible
        movie_response = query_gpt(user_input, session)
        # Try to extract the main movie discussed from the response (simple heuristic)
        import re
        m = re.search(r"([A-Za-z0-9: '\-]+) \([0-9]{4}\)", movie_response)
        if m:
            session["last_movie"] = m.group(1).strip()
            session["last_movie_context"] = movie_response

    # After getting movie_response, update preferences if a movie was discussed
    # (insert after updating last_movie and last_movie_context)
    if session.get("last_movie_context"):
        # Try to extract movie info from last_movie_context (simple heuristic)
        import re
        info = {}
        ctx = session["last_movie_context"]
        m = re.search(r"(?P<title>[A-Za-z0-9: '\-]+) \((?P<year>[0-9]{4})\): (?P<desc>.*?)\. Director: (?P<director>[^.]*). Cast: (?P<cast>[^.]*). Genre: (?P<genre>[^.]*).", ctx)
        if m:
            info = m.groupdict()
            info["cast"] = [a.strip() for a in info.get("cast", "").split(",") if a.strip()]
        update_preferences(session, info)

    session["chat_history"].append({"role": "assistant", "content": movie_response})
    return movie_response, session

# 🔷 Gradio callbacks
def on_submit(user_message, chat_history, state):
    response, state = handle_input(user_message, state)
    chat_history.append([user_message, response])
    return "", chat_history, state

def on_clear():
    return [], "", init_session()

def on_feedback(rating, chat_history, state):
    # Save feedback for the last exchange
    if chat_history and len(chat_history) > 0:
        last_user, last_response = chat_history[-1]
        user = state.get("name", "User")
        save_feedback(user, last_user, last_response, rating)
    return chat_history, state

# 🔷 Gradio app
with gr.Blocks() as demo:
    gr.Markdown("## \U0001F3AC Hollywood Movie Chatbot + User Info")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Message", placeholder="Hi!")
    clear = gr.Button("Clear Chat")
    state = gr.State(init_session())
    feedback = gr.Radio(["👍", "👎"], label="Rate the last answer", visible=True)

    msg.submit(on_submit, [msg, chatbot, state], [msg, chatbot, state])
    clear.click(on_clear, [], [chatbot, msg, state])
    feedback.change(on_feedback, [feedback, chatbot, state], [chatbot, state])

demo.launch()