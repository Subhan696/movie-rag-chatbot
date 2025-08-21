# -*- coding: utf-8 -*-
# 📦 Improved Hollywood Movie Chatbot (AstraDB + TMDB + Gemini)

import os
import time
import math
import csv
import re
import json
import difflib
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

# Vector search client
from astrapy import DataAPIClient

# Gemini
import google.generativeai as genai

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ----------------------------
# Env & SDKs
# ----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")
genai.configure(api_key=GEMINI_API_KEY)

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_API_URL = "https://api.themoviedb.org/3"

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT", "")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
if not ASTRA_DB_API_ENDPOINT or not ASTRA_DB_APPLICATION_TOKEN:
    raise RuntimeError("Missing ASTRA_DB_* env vars. Check .env")

client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
database = client.get_database(ASTRA_DB_API_ENDPOINT)
collection = database.get_collection("movies")

# ----------------------------
# Files
# ----------------------------
EXCEL_FILE = "user_info.xlsx"
FEEDBACK_FILE = "chatbot_feedback.csv"

# ----------------------------
# Embeddings
# ----------------------------
EMBEDDING_MODEL = "text-embedding-004"
VECTOR_DIMENSION = 768

def embed_text(text: str) -> Optional[List[float]]:
    """Get Gemini embedding vector for user query or document text."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        res = genai.embed_content(model=EMBEDDING_MODEL, content=text)
        emb = res.get("embedding") if isinstance(res, dict) else getattr(res, "embedding", None)
        if isinstance(emb, dict) and "values" in emb:
            return emb["values"]
        return emb
    except Exception as e:
        logging.exception("Embedding failed: %s", e)
        return None

# ----------------------------
# HTTP helpers (retry/backoff)
# ----------------------------
def http_get(url: str, params: Dict[str, Any], retries: int = 3, timeout: int = 10) -> Optional[requests.Response]:
    last_err = None
    for i in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            # Handle rate limits
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "2"))
                time.sleep(min(wait, 10))
                continue
            if 200 <= resp.status_code < 300:
                return resp
            last_err = f"HTTP {resp.status_code}: {resp.text[:150]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.5 * (i + 1))
    logging.warning("http_get failed for %s | params=%s | err=%s", url, params, last_err)
    return None

# ----------------------------
# Normalization & NLP
# ----------------------------
PLATFORMS = ['Netflix', 'Prime', 'Hulu', 'Disney', 'HBO', 'Max', 'Apple TV', 'Peacock', 'Paramount+']
GENRES = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
    'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
]

def normalize_title(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9: '\-]", " ", (t or "").strip())).strip().lower()

def clean_freeform_to_title(text: str) -> str:
    text = text or ""
    # Remove common prefixes
    text = re.sub(r"(?i)\b(tell me about|details of|info on|information on|who.*directed|cast of|where can i watch)\b", "", text)
    text = re.sub(r"(?i)\bmovie\b", "", text)
    text = text.strip()
    return text

def extract_year(text: str) -> Optional[str]:
    m = re.search(r"(19|20)\d{2}", text or "")
    return m.group(0) if m else None

def extract_platform(text: str) -> Optional[str]:
    for p in PLATFORMS:
        if p.lower() in (text or "").lower():
            # Normalize HBO/HBO Max to Max (TMDB nowadays uses "Max")
            return "Max" if p.lower().startswith("hbo") else p
    return None

def extract_genre(text: str) -> Optional[str]:
    for g in GENRES:
        if g.lower() in (text or "").lower():
            return g
    return None

def extract_titles(text: str) -> List[str]:
    text = text or ""
    # Quoted titles
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    titles = [q[0] or q[1] for q in quoted if q[0] or q[1]]

    # Capitalized phrases (loose heuristic)
    for phrase in re.findall(r'([A-Z][a-zA-Z0-9: \'\-]+)', text):
        if len(phrase.split()) > 1 and phrase not in titles:
            titles.append(phrase.strip())

    # Fallback to cleaned freeform
    cleaned = clean_freeform_to_title(text)
    if cleaned and cleaned not in titles and len(cleaned.split()) <= 6:
        titles.append(cleaned)

    # Dedup
    seen, out = set(), []
    for t in titles:
        n = normalize_title(t)
        if n and n not in seen:
            seen.add(n)
            out.append(t.strip())
    return out[:3]

# ----------------------------
# AstraDB retrieval & upsert
# ----------------------------
def retrieve_relevant_movies(user_query: str, top_k: int = 5) -> List[dict]:
    if not user_query:
        return []
    query_embedding = embed_text(user_query)
    if not isinstance(query_embedding, list):
        return []
    try:
        results = collection.find(
            {},
            sort={"$vector": query_embedding},
            limit=top_k,
            include_similarity=True
        )
        return list(results)
    except Exception as e:
        logging.exception("AstraDB error: %s", e)
        return []

def upsert_movie_into_astra(m: dict):
    """Upsert a movie doc (with vector) so next time Astra can hit it fast."""
    try:
        title = m.get("title") or ""
        year = str(m.get("year") or "")
        doc_id = f"{normalize_title(title)}_{year}"
        text_blob = format_movie_info(m)
        vec = embed_text(text_blob)
        doc = {
            "_id": doc_id,
            "title": title,
            "year": year,
            "description": m.get("description", ""),
            "director": m.get("director", ""),
            "cast": m.get("cast", []),
            "genre": m.get("genre", ""),
            "box_office": m.get("box_office", "Unknown"),
            "imdb_rating": m.get("imdb_rating", 0),
            "runtime_min": m.get("runtime_min", 0),
            "streaming": m.get("streaming", "Unknown"),
            "$vector": vec if isinstance(vec, list) else None,
        }
        # Remove None vector if embedding failed (still store doc)
        if doc["$vector"] is None:
            doc.pop("$vector", None)
        collection.update_one({"_id": doc_id}, {"$set": doc}, upsert=True)
    except Exception as e:
        logging.warning("Upsert into Astra failed: %s", e)

# ----------------------------
# TMDB fetchers (parallel & robust)
# ----------------------------
def tmdb_search_movie(query: str, year: Optional[str] = None) -> List[dict]:
    if not TMDB_API_KEY:
        return []
    url = f"{TMDB_API_URL}/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": query}
    if year:
        params["year"] = year
    resp = http_get(url, params)
    if not resp:
        return []
    return resp.json().get("results", []) or []

def tmdb_movie_details(movie_id: int) -> Optional[dict]:
    if not TMDB_API_KEY:
        return None
    url = f"{TMDB_API_URL}/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "append_to_response": "credits"}
    resp = http_get(url, params)
    if not resp:
        return None
    return resp.json()

def tmdb_watch_providers(movie_id: int, region: str = "US") -> dict:
    if not TMDB_API_KEY:
        return {}
    url = f"{TMDB_API_URL}/movie/{movie_id}/watch/providers"
    params = {"api_key": TMDB_API_KEY}
    resp = http_get(url, params)
    if not resp:
        return {}
    return (resp.json() or {}).get("results", {}).get(region, {}) or {}

def fetch_streaming_match(movie_ids: List[int], platform: Optional[str], region: str = "US") -> Optional[int]:
    """Return the first movie_id available on given platform (if provided)."""
    if not platform:
        return movie_ids[0] if movie_ids else None
    platform_norm = platform.lower()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(tmdb_watch_providers, mid, region): mid for mid in movie_ids}
        for fut in concurrent.futures.as_completed(futs):
            mid = futs[fut]
            providers = fut.result() or {}
            flatrate = providers.get("flatrate", []) or []
            if any(platform_norm in (p.get("provider_name", "").lower()) for p in flatrate):
                return mid
    return None

def pick_best_title(results: List[dict], target_title: str) -> dict:
    """Pick best TMDB result by fuzzy title match (and popularity as tie-breaker)."""
    if not results:
        return {}
    names = [r.get("title", "") for r in results]
    best = difflib.get_close_matches(target_title, names, n=1, cutoff=0.6)
    if best:
        name = best[0]
        for r in results:
            if r.get("title") == name:
                return r
    # fallback: most popular
    return sorted(results, key=lambda r: r.get("popularity", 0), reverse=True)[0]

def fetch_movie_from_tmdb(query: str, year: Optional[str] = None, platform: Optional[str] = None, region: str = "US") -> dict:
    if not TMDB_API_KEY:
        return {"error": "TMDB API key is missing. Please set TMDB_API_KEY in your .env file."}
    query = (query or "").strip()
    if not query:
        return {"error": "Empty query."}

    # Search
    results = tmdb_search_movie(query, year)
    if not results:
        # retry with cleaned query
        cleaned = clean_freeform_to_title(query)
        if cleaned != query:
            results = tmdb_search_movie(cleaned, year)
    if not results:
        return {"error": "No results found in TMDB."}

    # If platform is specified, find first result available there
    if platform:
        mid = fetch_streaming_match([r["id"] for r in results], platform, region=region)
        if not mid:
            return {"error": f"No movie found on {platform}."}
        base = next((r for r in results if r["id"] == mid), results[0])
    else:
        base = pick_best_title(results, query)

    details = tmdb_movie_details(base["id"])
    if not details:
        return {"error": "Failed to fetch details from TMDB."}

    # Parse details
    director = ""
    cast = []
    for member in (details.get("credits", {}) or {}).get("crew", []):
        if member.get("job") == "Director":
            director = member.get("name") or ""
            break
    for actor in (details.get("credits", {}) or {}).get("cast", [])[:5]:
        name = actor.get("name")
        if name:
            cast.append(name)

    # Providers (best effort, even if platform not specified)
    providers = tmdb_watch_providers(details["id"], region=region) or {}
    flatrate = providers.get("flatrate", []) or []
    stream_names = ", ".join(sorted({p.get("provider_name", "") for p in flatrate if p.get("provider_name")})) or "Unknown"

    movie = {
        "title": details.get("title", base.get("title", "N/A")),
        "year": (details.get("release_date", "") or "?")[:4] or "?",
        "description": details.get("overview", base.get("overview", "")) or "",
        "director": director,
        "cast": cast,
        "genre": ", ".join([g.get("name", "") for g in (details.get("genres") or []) if g.get("name")]),
        "box_office": details.get("revenue", "Unknown"),
        "imdb_rating": details.get("vote_average", 0),
        "runtime_min": details.get("runtime", 0),
        "streaming": stream_names
    }
    # Cache to Astra
    upsert_movie_into_astra(movie)
    return movie

def tmdb_discover_movies(year: Optional[str] = None, genre: Optional[str] = None, top_k: int = 5, region: str = "US") -> List[dict]:
    if not TMDB_API_KEY:
        return []
    genre_map = {
        'Action': 28, 'Adventure': 12, 'Animation': 16, 'Comedy': 35, 'Crime': 80, 'Documentary': 99,
        'Drama': 18, 'Family': 10751, 'Fantasy': 14, 'History': 36, 'Horror': 27, 'Music': 10402,
        'Mystery': 9648, 'Romance': 10749, 'Science Fiction': 878, 'TV Movie': 10770, 'Thriller': 53,
        'War': 10752, 'Western': 37
    }
    params = {"api_key": TMDB_API_KEY, "sort_by": "popularity.desc", "page": 1}
    if year:
        params["year"] = year
    if genre and genre_map.get(genre):
        params["with_genres"] = genre_map[genre]

    url = f"{TMDB_API_URL}/discover/movie"
    resp = http_get(url, params)
    if not resp:
        return []
    movies = (resp.json().get("results", []) or [])[:top_k]
    ids = [m["id"] for m in movies]

    # Parallel details
    def fetch(mid):
        d = tmdb_movie_details(mid)
        if not d:
            return None
        director = ""
        cast = []
        for member in (d.get("credits", {}) or {}).get("crew", []):
            if member.get("job") == "Director":
                director = member.get("name") or ""
                break
        for actor in (d.get("credits", {}) or {}).get("cast", [])[:5]:
            if actor.get("name"):
                cast.append(actor["name"])
        providers = tmdb_watch_providers(mid, region=region) or {}
        flatrate = providers.get("flatrate", []) or []
        stream_names = ", ".join(sorted({p.get("provider_name", "") for p in flatrate if p.get("provider_name")})) or "Unknown"
        movie = {
            "title": d.get("title", ""),
            "year": (d.get("release_date", "") or "?")[:4] or "?",
            "description": d.get("overview", ""),
            "director": director,
            "cast": cast,
            "genre": ", ".join([g.get("name", "") for g in (d.get("genres") or []) if g.get("name")]),
            "box_office": d.get("revenue", "Unknown"),
            "imdb_rating": d.get("vote_average", 0),
            "runtime_min": d.get("runtime", 0),
            "streaming": stream_names
        }
        upsert_movie_into_astra(movie)
        return movie

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        details = list(ex.map(fetch, ids))

    return [m for m in details if m]

# ----------------------------
# Formatting
# ----------------------------
def format_movie_info(movie: dict) -> str:
    return (
        f"{movie.get('title','N/A')} ({movie.get('year','?')}): {movie.get('description','').strip()} "
        f"Director: {movie.get('director','')}. Cast: {', '.join(movie.get('cast', [])) if isinstance(movie.get('cast'), list) else movie.get('cast','')}. "
        f"Genre: {movie.get('genre','')}. Box Office: {movie.get('box_office','Unknown')}. "
        f"IMDb: {round(float(movie.get('imdb_rating',0)), 1) if movie.get('imdb_rating') else 0}. "
        f"Available on: {movie.get('streaming','Unknown')}. Length: {movie.get('runtime_min',0)} minutes."
    )

# ----------------------------
# Session / State
# ----------------------------
def init_session():
    return {
        "name": None, "phone": None, "email": None, "location": None,
        "collected": False,
        "chat_history": [],
        "first_prompt": True,
        "last_movie": None,
        "last_movie_doc": None,   # full dict for follow-ups
        "preferences": {"genres": [], "actors": [], "directors": []},  # use lists (Gradio-safe)
        "region": "US",  # default; could infer from location later
        "recommendation_note": ""
    }

# ----------------------------
# Save user info & feedback
# ----------------------------
def save_user_info(session):
    user_info = {
        "Name": session.get("name"),
        "Phone": session.get("phone"),
        "Email": session.get("email"),
        "Location": session.get("location")
    }
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
        df = pd.concat([df, pd.DataFrame([user_info])], ignore_index=True)
    else:
        df = pd.DataFrame([user_info])
    df.to_excel(EXCEL_FILE, index=False)

def save_feedback(user, message, response, rating, comment=None):
    row = {
        "User": user,
        "Message": message,
        "Response": response,
        "Rating": rating,
        "Comment": comment or ""
    }
    file_exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ----------------------------
# LLM query (Gemini) with RAG context
# ----------------------------
def query_llm(user_query: str, session: dict, context_blocks: List[str]) -> str:
    user_name = session.get("name", "there")
    user_location = session.get("location", "your area")

    system_message = (
        f"You are a friendly assistant about Hollywood movies. "
        f"The user is {user_name} from {user_location}. Keep responses conversational and concise. "
        f"Use ONLY the provided context for facts unless asked general knowledge."
    )

    recent_history = session["chat_history"][-5:]
    history_text = ""
    for turn in recent_history:
        role = turn["role"].capitalize()
        history_text += f"{role}: {turn['content']}\n"

    movie_context = "\n".join(context_blocks) if context_blocks else "(No relevant context.)"

    prompt = (
        f"{system_message}\n\n"
        f"### CHAT HISTORY:\n{history_text}\n"
        f"### CONTEXT:\n{movie_context}\n"
        f"### USER QUESTION:\n{user_query}\n"
        f"Answer helpfully. If the user asks a follow-up like runtime/streaming/director, extract it directly from context."
    )

    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash-001")
        response = model.generate_content(prompt)
        return (response.text or "").strip()
    except Exception as e:
        logging.warning("Gemini error: %s", e)
        return "Sorry, I had trouble generating a response. Try again?"

# ----------------------------
# Preferences
# ----------------------------
def safe_add(lst: List[str], items: List[str]):
    s = set(lst)
    for x in items:
        x = (x or "").strip()
        if x and x not in s:
            lst.append(x); s.add(x)

def update_preferences(session, movie_info: dict):
    if not movie_info:
        return
    prefs = session.setdefault("preferences", {"genres": [], "actors": [], "directors": []})
    genres = movie_info.get("genre", "")
    if genres:
        safe_add(prefs["genres"], [g.strip() for g in genres.split(",") if g.strip()])
    safe_add(prefs["actors"], movie_info.get("cast", []))
    if movie_info.get("director"):
        safe_add(prefs["directors"], [movie_info["director"]])

def recommend_movies(session, top_k=3):
    # Try Astra based on preferences; fallback TMDB discover
    prefs = session.get("preferences", {})
    query_parts = []
    if prefs.get("genres"):
        query_parts.append("genre: " + ", ".join(prefs["genres"]))
    if prefs.get("actors"):
        query_parts.append("cast: " + ", ".join(prefs["actors"]))
    if prefs.get("directors"):
        query_parts.append("director: " + ", ".join(prefs["directors"]))
    query = ", ".join(query_parts) if query_parts else "popular movies"

    results = retrieve_relevant_movies(query, top_k=top_k) or []
    results = [r for r in results if isinstance(r, dict)]
    if len(results) < top_k:
        tmdb_more = tmdb_discover_movies(top_k=top_k, region=session.get("region", "US"))
        # Dedup by title+year
        seen = set((str(m.get('title','')).lower(), str(m.get('year',''))) for m in results)
        for m in tmdb_more:
            key = (str(m.get('title','')).lower(), str(m.get('year','')))
            if key not in seen:
                results.append(m)
                seen.add(key)
        session['recommendation_note'] = "Here are some popular picks:"
    else:
        session['recommendation_note'] = "Recommendations based on your taste:"
    return results[:top_k]

# ----------------------------
# Intent: follow-up detection
# ----------------------------
FOLLOWUP_REGEX = re.compile(
    r"(who (directed|is the director)|who (starred|acted)|cast|box office|genre|how long|runtime|where can i (watch|stream)|on (netflix|prime|hulu|max|disney|peacock|apple tv))",
    re.I
)

def is_followup_intent(user_input: str) -> bool:
    if FOLLOWUP_REGEX.search(user_input or ""):
        return True
    # quick pronoun heuristic
    if re.search(r"\b(it|this|that)\b", user_input or "", re.I):
        return True
    return False

# ----------------------------
# Input handler
# ----------------------------
def handle_input(user_input: str, session: dict) -> Tuple[str, dict]:
    user_input = (user_input or "").strip()
    user_input_lower = user_input.lower()

    # Allow info reset
    if user_input_lower in ["update my info", "change my info", "edit my info"]:
        session.update(init_session())
        session["first_prompt"] = False  # jump straight into collection
        return "Let's update your information. May I have your name, please?", session

    if session["first_prompt"]:
        session["first_prompt"] = False
        return (
            "Hello! 👋\nMay I have your name, please?\n\n"
            "*Privacy Notice: Your name, phone, email, and location are stored locally for personalization. "
            "You can skip phone/email/location by typing 'skip'.*",
            session
        )

    # Collect user info
    if not session["collected"]:
        if not session["name"]:
            if user_input_lower == "skip" or not user_input:
                return "Name is required. Please enter your name:", session
            session["name"] = user_input
            return "Thanks! Your phone number? (or type 'skip')", session
        elif not session["phone"]:
            if user_input_lower == "skip":
                session["phone"] = "(skipped)"
            else:
                if not re.match(r"^\+?\d{7,15}$", user_input):
                    return "That doesn't look like a valid phone. Enter a valid phone or type 'skip':", session
                session["phone"] = user_input
            return "Great. Your email address? (or type 'skip')", session
        elif not session["email"]:
            if user_input_lower == "skip":
                session["email"] = "(skipped)"
            else:
                if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", user_input):
                    return "That doesn't look like a valid email. Enter a valid email or type 'skip':", session
                session["email"] = user_input
            return "Lastly, your location (city/country)? (or type 'skip')", session
        elif not session["location"]:
            session["location"] = "(skipped)" if user_input_lower == "skip" else user_input
            # Optional: light region inference (very naive)
            if isinstance(session["location"], str) and "india" in session["location"].lower():
                session["region"] = "IN"
            save_user_info(session)
            session["collected"] = True
            return (
                f"Awesome, {session['name']} from {session['location']}! 🎉\n"
                f"Ask me about movies — cast, director, runtime, streaming, or get recommendations. 🍿",
                session
            )

    # --- Conversation / RAG ---
    session["chat_history"].append({"role": "user", "content": user_input})

    # Follow-up handling
    if is_followup_intent(user_input) and session.get("last_movie_doc"):
        # Answer directly from last movie doc
        m = session["last_movie_doc"]
        # If question is about streaming on specific platform, say yes/no
        p = extract_platform(user_input)
        if p:
            available = [s.strip().lower() for s in (m.get("streaming","") or "").split(",")]
            msg = f"Yes, it's on {p}." if p.lower() in available else f"I don't see {p} in its providers. Available on: {m.get('streaming','Unknown')}."
            session["chat_history"].append({"role": "assistant", "content": msg})
            return msg, session
        # Other follow-ups → compose a concise answer
        snippet = []
        if re.search(r"who (directed|is the director)", user_input, re.I) and m.get("director"):
            snippet.append(f"Director: {m['director']}")
        if re.search(r"(who (starred|acted)|cast)", user_input, re.I) and m.get("cast"):
            snippet.append(f"Cast: {', '.join(m['cast'])}")
        if re.search(r"(box office)", user_input, re.I) and m.get("box_office") not in (None, "Unknown"):
            snippet.append(f"Box office: {m['box_office']}")
        if re.search(r"(genre)", user_input, re.I) and m.get("genre"):
            snippet.append(f"Genre: {m['genre']}")
        if re.search(r"(how long|runtime)", user_input, re.I) and m.get("runtime_min"):
            snippet.append(f"Runtime: {m['runtime_min']} min")
        if re.search(r"(where can i (watch|stream))", user_input, re.I):
            snippet.append(f"Streaming: {m.get('streaming','Unknown')}")
        if snippet:
            msg = f"{m['title']} ({m['year']}): " + " | ".join(snippet)
            session["chat_history"].append({"role": "assistant", "content": msg})
            return msg, session
        # If we can't parse, fall back to LLM with context
        ctx = [format_movie_info(m)]
        msg = query_llm(user_input, session, ctx)
        session["chat_history"].append({"role": "assistant", "content": msg})
        return msg, session

    # New query path
    year = extract_year(user_input)
    platform = extract_platform(user_input)
    genre = extract_genre(user_input)
    titles = extract_titles(user_input)

    context_blocks = []
    region = session.get("region", "US")

    if titles:
        # Try Astra hit first by exact title
        for t in titles:
            # Try Astra
            astra_hits = retrieve_relevant_movies(t, top_k=3)
            exact = None
            for m in astra_hits:
                if normalize_title(m.get("title","")) == normalize_title(t):
                    exact = m; break
            if exact:
                context_blocks.append(format_movie_info(exact))
                session["last_movie_doc"] = {
                    "title": exact.get("title"), "year": exact.get("year"),
                    "description": exact.get("description",""), "director": exact.get("director",""),
                    "cast": exact.get("cast",[]), "genre": exact.get("genre",""),
                    "box_office": exact.get("box_office","Unknown"), "imdb_rating": exact.get("imdb_rating",0),
                    "runtime_min": exact.get("runtime_min",0), "streaming": exact.get("streaming","Unknown")
                }
                update_preferences(session, session["last_movie_doc"])
                break
            # Else TMDB
            tm = fetch_movie_from_tmdb(t, year=year, platform=platform, region=region)
            if "error" not in tm:
                context_blocks.append(format_movie_info(tm))
                session["last_movie_doc"] = tm
                update_preferences(session, tm)
                break

    if not context_blocks and (year or genre):
        # Discovery path
        # Try Astra with combined query
        astra_ctx = retrieve_relevant_movies(f"{genre or ''} {year or ''}".strip(), top_k=5)
        if astra_ctx:
            for m in astra_ctx:
                context_blocks.append(format_movie_info(m))
        else:
            discovered = tmdb_discover_movies(year=year, genre=genre, top_k=5, region=region)
            for m in discovered:
                context_blocks.append(format_movie_info(m))

    if not context_blocks and not titles:
        # Generic recommendations
        recs = recommend_movies(session, top_k=3)
        if recs:
            note = session.get("recommendation_note","Here are some movies you might like:")
            context_blocks.append(note)
            for m in recs:
                context_blocks.append(f"- {m.get('title','N/A')} ({m.get('year','?')}): { (m.get('description','') or '')[:110]}...")

    if not context_blocks:
        msg = "I couldn’t find a good match. Try giving a movie title, year, genre, or where you want to stream it."
        session["chat_history"].append({"role": "assistant", "content": msg})
        return msg, session

    # Generate final answer
    answer = query_llm(user_input, session, context_blocks)
    session["chat_history"].append({"role": "assistant", "content": answer})
    return answer, session

# ----------------------------
# Gradio Callbacks
# ----------------------------
def on_submit(user_message, chat_history, state):
    response, state = handle_input(user_message, state)
    chat_history.append([user_message, response])
    return "", chat_history, state

def on_clear():
    return [], "", init_session()

def on_feedback(rating, comment, chat_history, state):
    if chat_history:
        last_user, last_response = chat_history[-1]
        user = state.get("name", "User")
        save_feedback(user, last_user, last_response, rating, comment)
    return chat_history, state

# ----------------------------
# UI
# ----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🎬 Hollywood Movie Chatbot + User Info (Improved)")
    chatbot = gr.Chatbot(height=420)
    with gr.Row():
        msg = gr.Textbox(label="Your Message", placeholder="Try: where can I stream The Dark Knight (2008)?")
    with gr.Row():
        clear = gr.Button("Clear Chat")
        feedback = gr.Radio(["👍", "👎"], label="Rate the last answer")
        feedback_comment = gr.Textbox(label="Feedback Comment (optional)", lines=2)

    state = gr.State(init_session())

    msg.submit(on_submit, [msg, chatbot, state], [msg, chatbot, state])
    clear.click(on_clear, [], [chatbot, msg, state])
    feedback.change(on_feedback, [feedback, feedback_comment, chatbot, state], [chatbot, state])
    feedback_comment.submit(on_feedback, [feedback, feedback_comment, chatbot, state], [chatbot, state])

demo.launch()
