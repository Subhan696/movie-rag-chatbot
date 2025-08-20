# 📦 Import necessary libraries
import os
import pandas as pd
import google.generativeai as genai  # ✅ Gemini SDK
from dotenv import load_dotenv
import gradio as gr
from typing import List, Dict, Any
import requests  # For TMDB API
import csv
import re

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

def retrieve_relevant_movies(user_query: str, top_k: int = 5) -> list:
    # Restore AstraDB logic for hybrid approach
    try:
        query_embedding = embed_text(user_query)
        if not isinstance(query_embedding, list):
            return []
        results = collection.find(
            {},
            sort={"$vector": query_embedding},
            limit=top_k,
            include_similarity=True,
        )
        return list(results)
    except Exception:
        return []

def extract_movie_titles(text):
    """Extract likely movie titles from user input using capitalization, known patterns, and TMDB fuzzy search as fallback."""
    # Look for quoted titles
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    titles = [q[0] or q[1] for q in quoted if q[0] or q[1]]
    # Also look for capitalized phrases (simple heuristic)
    cap_phrases = re.findall(r'([A-Z][a-zA-Z0-9: \'\-]+)', text)
    for phrase in cap_phrases:
        if len(phrase.split()) > 1 and phrase not in titles:
            titles.append(phrase.strip())
    # Remove duplicates, preserve order
    seen = set()
    result = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            result.append(t)
    # Fallback: If nothing found, try TMDB search for fuzzy match
    if not result and TMDB_API_KEY:
        search_url = f"{TMDB_API_URL}/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": text}
        try:
            resp = requests.get(search_url, params=params)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    # Return top 1-2 matches
                    for m in results[:2]:
                        title = m.get("title")
                        if title and title not in result:
                            result.append(title)
        except Exception:
            pass
    return result

def extract_year_and_platform(text):
    """Extract year and streaming platform from user input."""
    year = None
    platform = None
    # Look for a 4-digit year
    m = re.search(r'(19|20)\d{2}', text)
    if m:
        year = m.group(0)
    # Look for common streaming platforms
    platforms = ['Netflix', 'Prime', 'Hulu', 'Disney', 'HBO', 'Apple TV', 'Peacock']
    for p in platforms:
        if p.lower() in text.lower():
            platform = p
            break
    return year, platform

def extract_genre(text):
    """Extract a likely genre from user input."""
    # List of common genres in TMDB
    genres = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
        'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
    ]
    for g in genres:
        if g.lower() in text.lower():
            return g
    return None

# TMDB API setup
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_API_URL = "https://api.themoviedb.org/3"

def fetch_movie_from_tmdb(query: str, year: str = None, platform: str = None) -> dict:
    """Fetch movie details from TMDB by search query, with optional year and platform filtering."""
    if not TMDB_API_KEY:
        return {}
    search_url = f"{TMDB_API_URL}/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": query}
    if year:
        params["year"] = year
    resp = requests.get(search_url, params=params)
    if resp.status_code != 200:
        return {}
    results = resp.json().get("results", [])
    if not results:
        return {}
    # If platform is specified, filter by streaming provider (using TMDB watch/providers API)
    if platform:
        for movie in results:
            prov_url = f"{TMDB_API_URL}/movie/{movie['id']}/watch/providers"
            prov_params = {"api_key": TMDB_API_KEY}
            prov_resp = requests.get(prov_url, params=prov_params)
            if prov_resp.status_code == 200:
                prov_data = prov_resp.json().get("results", {})
                # Check for US region (can be adjusted)
                us = prov_data.get("US", {})
                flatrate = us.get("flatrate", [])
                if any(platform.lower() in p.get("provider_name", "").lower() for p in flatrate):
                    # Add streaming info
                    movie["streaming"] = platform
                    break
        else:
            # No movie found on that platform
            return {}
        # Use the first matching movie
        movie = movie
    else:
        movie = results[0]
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
        "streaming": movie.get("streaming", "Unknown")
    }

def tmdb_discover_movies(year=None, genre=None, platform=None, top_k=5):
    """Fetch movies from TMDB discover endpoint with optional year, genre, and platform filters."""
    if not TMDB_API_KEY:
        return []
    discover_url = f"{TMDB_API_URL}/discover/movie"
    params = {"api_key": TMDB_API_KEY, "sort_by": "popularity.desc", "page": 1}
    if year:
        params["year"] = year
    if genre:
        # Map genre name to TMDB genre ID
        genre_map = {
            'Action': 28, 'Adventure': 12, 'Animation': 16, 'Comedy': 35, 'Crime': 80, 'Documentary': 99,
            'Drama': 18, 'Family': 10751, 'Fantasy': 14, 'History': 36, 'Horror': 27, 'Music': 10402,
            'Mystery': 9648, 'Romance': 10749, 'Science Fiction': 878, 'TV Movie': 10770, 'Thriller': 53,
            'War': 10752, 'Western': 37
        }
        genre_id = genre_map.get(genre)
        if genre_id:
            params["with_genres"] = genre_id
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

    context_blocks = []
    year, platform = extract_year_and_platform(user_query)
    genre = extract_genre(user_query)
    movie_titles = extract_movie_titles(user_query)
    # If explicit movie titles, try AstraDB first, then TMDB if missing/empty
    if movie_titles:
        for title in movie_titles:
            found_in_db = None
            for m in retrieve_relevant_movies(title, top_k=1):
                if title.lower() == str(m.get('title', '')).lower():
                    found_in_db = m
                    break
            # Check if DB result is missing key info
            missing_info = (
                not found_in_db or
                not found_in_db.get('description') or
                not found_in_db.get('director') or
                not found_in_db.get('cast')
            )
            if missing_info:
                tmdb_movie = fetch_movie_from_tmdb(title, year=year, platform=platform)
                if tmdb_movie:
                    context_blocks.append(format_movie_info(tmdb_movie))
                    continue
            if found_in_db:
                context_blocks.append(format_movie_info(found_in_db))
    # If no explicit titles but year/genre present, use AstraDB first, then TMDB discover
    if not context_blocks and (year or genre):
        db_results = retrieve_relevant_movies(f"{genre or ''} {year or ''}", top_k=5)
        if db_results:
            for m in db_results:
                context_blocks.append(format_movie_info(m))
        else:
            discovered = tmdb_discover_movies(year=year, genre=genre, platform=platform, top_k=5)
            if discovered:
                context_blocks.append("Here are some movies I found:")
                for m in discovered:
                    context_blocks.append(f"- {m['title']} ({m['year']}): {m['description'][:100]}...")
    if not context_blocks:
        context_blocks.append("(No relevant movies found in DB or TMDB for your query.)")
    movie_context = "\n".join(context_blocks)

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

# --- Constants for genres and platforms ---
GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
    'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
]
PLATFORMS = ['Netflix', 'Prime', 'Hulu', 'Disney', 'HBO', 'Apple TV', 'Peacock']

# --- Utility for formatting movie info ---
def format_movie_info(movie):
    return (
        f"{movie.get('title','N/A')} ({movie.get('year','?')}): {movie.get('description','')}. "
        f"Director: {movie.get('director','')}. Cast: {', '.join(movie.get('cast', [])) if isinstance(movie.get('cast'), list) else movie.get('cast','')}. "
        f"Genre: {movie.get('genre','')}. Box Office: {movie.get('box_office','Unknown')}. "
        f"IMDb: {movie.get('imdb_rating',0)}. Available on: {movie.get('streaming','Unknown')}. "
        f"Length: {movie.get('runtime_min',0)} minutes."
    )

def recommend_movies(session, top_k=3):
    # Hybrid: Try AstraDB, then TMDB discover
    prefs = session.get("preferences", {})
    query_parts = []
    if prefs.get("genres"):
        query_parts.append(f"genre: {', '.join(list(prefs['genres']))}")
    if prefs.get("actors"):
        query_parts.append(f"cast: {', '.join(list(prefs['actors']))}")
    if prefs.get("directors"):
        query_parts.append(f"director: {', '.join(list(prefs['directors']))}")
    query = ", ".join(query_parts) if query_parts else "popular movies"
    try:
        results = retrieve_relevant_movies(query, top_k=top_k)
    except Exception:
        results = []
    # Fallback: TMDB discover if not enough results
    if not results or len(results) < top_k:
        tmdb_results = tmdb_discover_movies(top_k=top_k)
        # Deduplicate by title+year
        seen = set((str(m.get('title','')).lower(), str(m.get('year',''))) for m in results)
        deduped = results[:]
        for m in tmdb_results:
            key = (str(m.get('title','')).lower(), str(m.get('year','')))
            if key not in seen:
                deduped.append(m)
                seen.add(key)
        # Add explanation for fallback
        if not results:
            session['recommendation_note'] = "I couldn't find enough matches in my database, so here are some popular movies from TMDB:"
        else:
            session['recommendation_note'] = "Here are some additional movies from TMDB:"
        return deduped[:top_k]
    session['recommendation_note'] = "Here are some recommendations based on your preferences:"
    return results

def is_valid_email(email):
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email)

def is_valid_phone(phone):
    return re.match(r"^\+?\d{7,15}$", phone)

def is_followup_intent(user_input, session):
    """Use Gemini to classify if the user input is a follow-up/ambiguous query."""
    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash-001")
        prompt = (
            "You are an intent classifier for a movie chatbot. "
            "Given the user's message and recent chat history, answer only 'yes' if the message is a follow-up (e.g., refers to 'it', 'that', 'this', or asks about director, cast, box office, genre, runtime, or streaming info of a previously discussed movie), otherwise answer 'no'.\n"
            f"User message: {user_input}\n"
            f"Recent chat history: {session['chat_history'][-3:] if session.get('chat_history') else ''}"
        )
        response = model.generate_content(prompt)
        return response.text.strip().lower().startswith('yes')
    except Exception:
        # Fallback to regex if Gemini fails
        followup_patterns = [
            r"who (directed|is the director)",
            r"who (starred|acted|is in it)",
            r"what (is|was) the box office",
            r"what (is|was) the genre",
            r"how long (is|was) it",
            r"where can i watch|stream (it|this|that)",
            r"(it|this|that)"
        ]
        return any(re.search(pat, user_input, re.I) for pat in followup_patterns)

# 🔷 Handle input from user
def handle_input(user_input, session):
    user_input_lower = user_input.strip().lower()
    # Allow user to update info at any time
    if user_input_lower in ["update my info", "change my info", "edit my info"]:
        session["collected"] = False
        session["name"] = None
        session["phone"] = None
        session["email"] = None
        session["location"] = None
        return "Let's update your information. May I have your name, please?", session

    if session["first_prompt"]:
        session["first_prompt"] = False
        return (
            "Hello! \U0001F44B\nMay I have your name, please?\n\n"
            "*Privacy Notice: Your name, phone, email, and location are stored in a local file for personalization. "
            "You may skip phone/email/location by typing 'skip'.*\n",
            session
        )

    if not session["collected"]:
        if not session["name"]:
            if user_input_lower == "skip" or not user_input.strip():
                return "Name is required. Please enter your name:", session
            session["name"] = user_input.strip()
            return "Thank you, may I have your phone number? (or type 'skip')", session
        elif not session["phone"]:
            if user_input_lower == "skip":
                session["phone"] = "(skipped)"
                return "Great. May I have your email address? (or type 'skip')", session
            elif not is_valid_phone(user_input.strip()):
                return "That doesn't look like a valid phone number. Please enter a valid phone or type 'skip':", session
            session["phone"] = user_input.strip()
            return "Great. May I have your email address? (or type 'skip')", session
        elif not session["email"]:
            if user_input_lower == "skip":
                session["email"] = "(skipped)"
                return "Thank you. Lastly, may I know your location? (or type 'skip')", session
            elif not is_valid_email(user_input.strip()):
                return "That doesn't look like a valid email address. Please enter a valid email or type 'skip':", session
            session["email"] = user_input.strip()
            return "Thank you. Lastly, may I know your location? (or type 'skip')", session
        elif not session["location"]:
            if user_input_lower == "skip":
                session["location"] = "(skipped)"
            else:
                session["location"] = user_input.strip()
            session["collected"] = True
            save_user_info(session)
            return (
                f"Awesome, {session['name']} from {session['location']}! \U0001F389\n"
                f"You can now ask me anything about Hollywood movies — cast, director, box office, or where to stream them. "
                f"Let's get started! \U0001F37F\n\n*You can update your info anytime by typing 'update my info'.*",
                session
            )

    # --- Context Awareness: Track last discussed movie ---
    session.setdefault("last_movie", None)
    session.setdefault("last_movie_context", None)

    # Add user input to chat history
    session["chat_history"].append({"role": "user", "content": user_input})

    # Improved follow-up detection
    is_followup = is_followup_intent(user_input, session)
    if is_followup and session["last_movie_context"]:
        # Use last movie context for the answer
        movie_response = session["last_movie_context"]
    else:
        # Normal flow: get response and update last_movie if possible
        movie_response = query_gpt(user_input, session)
        # Try to extract the main movie discussed from the response (simple heuristic)
        m = re.search(r"([A-Za-z0-9: '\-]+) \([0-9]{4}\)", movie_response)
        if m:
            session["last_movie"] = m.group(1).strip()
            session["last_movie_context"] = movie_response

    # After getting movie_response, update preferences if a movie was discussed
    if session.get("last_movie_context"):
        # Try to extract movie info from last_movie_context (simple heuristic)
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