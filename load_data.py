"""
load_data.py

Ingest Hollywood movies from TMDB (2000–2025, rating > 6.0) into AstraDB with
Gemini embeddings for vector search.

Environment variables (stored in .env):
- GEMINI_API_KEY
- TMDB_API_KEY
- ASTRA_DB_API_ENDPOINT
- ASTRA_DB_APPLICATION_TOKEN
- ASTRA_DB_KEYSPACE (optional for serverless; required for legacy DBs)

Run:
  python load_data.py

This script will:
- Fetch movies via TMDB discover API with retries and pagination
- Enrich with details (cast, director, runtime, revenue). Streaming is mocked
- Generate embeddings via Gemini for each movie
- Insert into AstraDB collection `movies` with vector field `embedding`
- Skip duplicates based on stable `_id` of `title_lower + '_' + year`
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, List, Optional, Generator
import time

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai
from astrapy.info import CollectionDefinition, CollectionVectorOptions
from astrapy.constants import VectorMetric

try:
    # astrapy >= 1.4
    from astrapy import DataAPIClient
except Exception as exc:  # pragma: no cover - import-time error messaging
    raise RuntimeError(
        "astrapy is required. Please install with: pip install astrapy"
    ) from exc


# ---------- Configuration ----------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT", "")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")  # optional on serverless

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in .env")
if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY is missing in .env")
if not ASTRA_DB_API_ENDPOINT or not ASTRA_DB_APPLICATION_TOKEN:
    raise RuntimeError("ASTRA_DB_API_ENDPOINT or ASTRA_DB_APPLICATION_TOKEN missing in .env")

genai.configure(api_key=GEMINI_API_KEY)

# text-embedding-004 returns 768-d vectors
EMBEDDING_MODEL = "text-embedding-004"
VECTOR_DIMENSION = 768
VECTOR_METRIC = "cosine"

TMDB_BASE_URL = "https://api.themoviedb.org/3"
HEADERS_TMDB = {"Authorization": f"Bearer {TMDB_API_KEY}"} if TMDB_API_KEY and TMDB_API_KEY.startswith("eyJ") else {}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------- Astra helpers ----------
def get_astra_collection(collection_name: str):
    """Connect to Astra and create/get a vector-enabled collection, waiting until ready.

    Respects `ASTRA_DB_KEYSPACE` when provided.
    """
    client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
    database = client.get_database(ASTRA_DB_API_ENDPOINT)

    created_now = False
    keyspace = ASTRA_DB_KEYSPACE
    if keyspace:
        logging.info(f"Using keyspace='{keyspace}' for collection '{collection_name}'")
    else:
        logging.info("No ASTRA_DB_KEYSPACE provided; using database default keyspace")
    # Pre-check what exists
    try:
        existing = database.list_collections(keyspace=keyspace) if keyspace else database.list_collections()
        # list_collections may return list[str] or list[dict]
        if existing and isinstance(existing[0], dict):
            existing = [c.get("name") for c in existing if c.get("name")]
        logging.info(f"Existing collections in keyspace: {existing}")
    except Exception as e:
        logging.debug(f"list_collections failed before create: {e}")
        existing = []

    if collection_name not in existing:
        try:
            definition = CollectionDefinition(
                vector=CollectionVectorOptions(
                    dimension=VECTOR_DIMENSION,
                    metric=VectorMetric.COSINE,
                )
            )
            if keyspace:
                database.create_collection(collection_name, definition=definition, keyspace=keyspace)
            else:
                database.create_collection(collection_name, definition=definition)
            created_now = True
            logging.info(
                f"Requested creation of collection '{collection_name}' (dim={VECTOR_DIMENSION}, metric={VECTOR_METRIC})"
            )
        except Exception as e:
            # Surface permissions/misconfig immediately
            raise RuntimeError(
                "Failed to create collection. Ensure your ASTRA_DB_APPLICATION_TOKEN has Data API "
                "permissions (Vector Search/Document Editor or higher) and endpoint/keyspace are correct.\n"
                f"Details: {e}"
            ) from e

    # If not created just now, verify existence via list_collections
    try:
        coll_names = database.list_collections(keyspace=keyspace) if keyspace else database.list_collections()
        if coll_names and isinstance(coll_names[0], dict):
            coll_names = [c.get("name") for c in coll_names if c.get("name")]
        if collection_name not in coll_names:
            # Wait for it to appear
            logging.info("Waiting for collection to appear in list...")
            wait_start = time.time()
            while time.time() - wait_start < 30:
                try:
                    coll_names = database.list_collections(keyspace=keyspace) if keyspace else database.list_collections()
                    if coll_names and isinstance(coll_names[0], dict):
                        coll_names = [c.get("name") for c in coll_names if c.get("name")]
                    if collection_name in coll_names:
                        break
                except Exception:
                    pass
                time.sleep(1)
            if collection_name not in coll_names:
                raise RuntimeError("Collection did not appear in list_collections within timeout")
    except Exception as e:
        logging.debug(f"list_collections post-create failed (will continue): {e}")

    coll = database.get_collection(collection_name, keyspace=keyspace) if keyspace else database.get_collection(collection_name)

    # Wait until the collection is queryable to avoid race on first inserts
    max_wait_seconds = 30
    start_time = time.time()
    while True:
        try:
            _ = list(coll.find({}, limit=1))
            if created_now:
                logging.info(f"Collection '{collection_name}' is ready")
            break
        except Exception as e:
            if "COLLECTION_NOT_EXIST" in str(e) or "does not exist" in str(e).lower():
                if time.time() - start_time > max_wait_seconds:
                    raise RuntimeError(
                        f"Timed out waiting for collection '{collection_name}' to be ready"
                    ) from e
                time.sleep(1)
                continue
            raise

    return coll


# ---------- Embeddings ----------
def generate_embedding(text: str) -> List[float]:
    """Generate a Gemini embedding for given text.

    Handles minor response shape variations between SDK versions.
    """
    text = (text or "").strip()
    if not text:
        return [0.0] * VECTOR_DIMENSION

    result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
    # Possible response shapes: {"embedding": [...] } or {"embedding": {"values": [...]}}
    embedding = None
    if isinstance(result, dict):
        emb = result.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            embedding = emb["values"]
        else:
            embedding = emb
    else:
        # Some SDKs return an object with .embedding
        emb = getattr(result, "embedding", None)
        if isinstance(emb, dict) and "values" in emb:
            embedding = emb["values"]
        else:
            embedding = emb

    if not isinstance(embedding, list):
        raise RuntimeError("Unexpected embedding response shape from Gemini")
    return embedding


# ---------- TMDB fetchers with retries ----------
class TransientHTTPError(Exception):
    pass


def _check_response(resp: requests.Response) -> None:
    if resp.status_code >= 500:
        raise TransientHTTPError(f"TMDB server error: {resp.status_code}")
    if resp.status_code != 200:
        raise RuntimeError(f"TMDB error: {resp.status_code} - {resp.text[:200]}")


@retry(
    reraise=True,
    retry=retry_if_exception_type(TransientHTTPError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{TMDB_BASE_URL}{path}"
    if HEADERS_TMDB:
        resp = requests.get(url, headers=HEADERS_TMDB, params=params, timeout=30)
    else:
        # API key via query param
        params = {**params, "api_key": TMDB_API_KEY}
        resp = requests.get(url, params=params, timeout=30)
    _check_response(resp)
    return resp.json()


def discover_movies(year_start: int = 2000, year_end: int = 2025, min_rating: float = 6.0, max_pages_per_year: int = 5) -> Generator[Dict[str, Any], None, None]:
    """Yield TMDB discover results within year range with pagination.

    Limits pages per year to avoid huge runs. Adjust as needed.
    """
    for year in range(year_start, year_end + 1):
        for page in range(1, max_pages_per_year + 1):
            data = tmdb_get(
                "/discover/movie",
                {
                    "language": "en-US",
                    "sort_by": "popularity.desc",
                    "include_adult": "false",
                    "include_video": "false",
                    "page": page,
                    "primary_release_year": year,
                    "vote_average.gte": min_rating,
                    "with_original_language": "en",
                    "region": "US",
                },
            )
            results = data.get("results", [])
            if not results:
                break
            for item in results:
                yield item


@retry(
    reraise=True,
    retry=retry_if_exception_type(TransientHTTPError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def fetch_movie_details(movie_id: int) -> Optional[Dict[str, Any]]:
    params = {"append_to_response": "credits,watch/providers"}
    data = tmdb_get(f"/movie/{movie_id}", params)
    return data


def extract_movie_document(details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map TMDB detail payload to our AstraDB movie document schema."""
    title = (details.get("title") or details.get("name") or "").strip()
    release_date = (details.get("release_date") or "").strip()
    if not title or not release_date:
        return None
    try:
        year = int(release_date.split("-")[0])
    except Exception:
        return None

    overview = (details.get("overview") or "").strip()

    # Director
    director = ""
    for crew in (details.get("credits", {}).get("crew", []) or []):
        if (crew.get("job") or "").lower() == "director":
            director = crew.get("name", "").strip()
            break

    # Cast (top 5 names)
    cast_names: List[str] = []
    for member in (details.get("credits", {}).get("cast", []) or [])[:5]:
        name = member.get("name")
        if name:
            cast_names.append(name.strip())

    # Genre (first genre name)
    genres = details.get("genres", []) or []
    genre_name = ", ".join([g.get("name", "").strip() for g in genres if g.get("name")]) or "Unknown"

    # Box office
    revenue = details.get("revenue") or 0
    box_office = f"${revenue:,}" if isinstance(revenue, int) and revenue > 0 else "Unknown"

    # TMDB rating used as IMDb proxy (field named imdb_rating as per requirement)
    imdb_rating = float(details.get("vote_average") or 0.0)

    # Streaming availability (mocked if not available)
    streaming = "Unknown"
    providers = details.get("watch/providers", {}).get("results", {})
    us_providers = providers.get("US", {}) if isinstance(providers, dict) else {}
    flatrate = us_providers.get("flatrate") or []
    if flatrate:
        names = [p.get("provider_name", "").strip() for p in flatrate if p.get("provider_name")]
        streaming = ", ".join([n for n in names if n]) or "Unknown"
    if not streaming or streaming == "Unknown":
        # Mock if nothing found
        streaming = "Check Netflix/Prime/Hulu"

    runtime_min = int(details.get("runtime") or 0)

    # Build embedding text
    embedding_text = (
        f"Title: {title}. Year: {year}. Genre: {genre_name}. "
        f"Director: {director}. Cast: {', '.join(cast_names)}. "
        f"Streaming: {streaming}. Rating: {imdb_rating}. "
        f"Description: {overview}"
    )
    embedding = generate_embedding(embedding_text)

    doc_id = f"{title.lower().strip()}_{year}"

    movie_doc = {
        "_id": doc_id,  # duplicate prevention id
        "title": title,
        "year": year,
        "description": overview,
        "director": director,
        "cast": cast_names,
        "genre": genre_name,
        "box_office": box_office,
        "imdb_rating": imdb_rating,
        "streaming": streaming,
        "runtime_min": runtime_min,
        "embedding": embedding,  # store for debugging/inspection
        "$vector": embedding,    # required by Astra Data API for vector search
    }
    return movie_doc


def insert_if_new(collection, movie_doc: Dict[str, Any]) -> bool:
    """Insert document if `_id` is not already present.

    Returns True if inserted, False if skipped as duplicate. Logs action with title/year.
    """
    existing = collection.find_one({"_id": movie_doc["_id"]})
    title = movie_doc.get("title", "Unknown")
    year = movie_doc.get("year", "?")
    if existing is not None:
        logging.info(f"Skipped duplicate: {title} ({year})")
        return False
    collection.insert_one(movie_doc)
    logging.info(f"Inserted: {title} ({year})")
    return True


def main() -> None:
    collection = get_astra_collection("movies")

    inserted = 0
    skipped = 0

    # Iterate discovered movies and enrich
    logging.info("Discovering movies from TMDB... this can take a few minutes on first run")
    items = list(discover_movies())
    logging.info(f"Discovered {len(items)} candidate movies from TMDB")

    for item in tqdm(items, desc="Ingesting movies"):
        movie_id = item.get("id")
        if not movie_id:
            continue

        try:
            details = fetch_movie_details(int(movie_id))
            if not details:
                continue
            movie_doc = extract_movie_document(details)
            if not movie_doc:
                continue
            if insert_if_new(collection, movie_doc):
                inserted += 1
            else:
                skipped += 1
        except Exception as exc:
            logging.warning(f"Skipping movie id={movie_id} due to error: {exc}")

    logging.info(f"Done. Inserted: {inserted}, Skipped (duplicates): {skipped}")


if __name__ == "__main__":
    main()


