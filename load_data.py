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
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Generator, Tuple
from queue import Queue
import signal

import requests
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    retry_if_result,
    RetryCallState
)
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai
from astrapy import DataAPIClient
from astrapy.info import CollectionDefinition, CollectionVectorOptions
from astrapy.constants import VectorMetric
from pydantic import BaseModel, Field, validator
from ratelimit import limits, sleep_and_retry

try:
    # astrapy >= 1.4
    from astrapy import DataAPIClient
except Exception as exc:  # pragma: no cover - import-time error messaging
    raise RuntimeError(
        "astrapy is required. Please install with: pip install astrapy"
    ) from exc


# ---------- Configuration ----------
class Settings(BaseModel):
    # API Keys
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    tmdb_api_key: str = Field(..., env="TMDB_API_KEY")
    astra_db_endpoint: str = Field(..., env="ASTRA_DB_API_ENDPOINT")
    astra_db_token: str = Field(..., env="ASTRA_DB_APPLICATION_TOKEN")
    astra_db_keyspace: Optional[str] = Field(None, env="ASTRA_DB_KEYSPACE")
    
    # Batch processing
    batch_size: int = 10
    max_workers: int = 5
    
    # Rate limiting
    tmdb_rate_limit: int = 40  # requests per 10 seconds
    gemini_rate_limit: int = 60  # requests per minute
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: int = 2  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Load settings
settings = Settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('movie_loader.log')
    ]
)
logger = logging.getLogger(__name__)

# Global state for graceful shutdown
shutdown_event = False

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    global shutdown_event
    logger.warning("Shutdown signal received. Finishing current batch...")
    shutdown_event = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ---------- Astra helpers ----------
def get_astra_collection(collection_name: str):
    """Connect to Astra and create/get a vector-enabled collection, waiting until ready.

    Respects `ASTRA_DB_KEYSPACE` when provided.
    """
    client = DataAPIClient(settings.astra_db_token)
    database = client.get_database(settings.astra_db_endpoint)

    created_now = False
    keyspace = settings.astra_db_keyspace
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
                    dimension=768,
                    metric=VectorMetric.COSINE,
                )
            )
            if keyspace:
                database.create_collection(collection_name, definition=definition, keyspace=keyspace)
            else:
                database.create_collection(collection_name, definition=definition)
            created_now = True
            logging.info(
                f"Requested creation of collection '{collection_name}' (dim=768, metric=COSINE)"
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time
from typing import Optional, List, Dict, Any
import logging

# Configuration
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_RETRY_WAIT_SECONDS = 2
EMBEDDING_RATE_LIMIT_DELAY = 0.1  # 100ms delay between API calls
LAST_API_CALL_TIME = 0

@retry(
    stop=stop_after_attempt(EMBEDDING_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
    reraise=True
)
def generate_embedding(text: str, model_name: str = "text-embedding-004") -> List[float]:
    """Generate a Gemini embedding for the given text with retries and rate limiting.

    Args:
        text: The input text to generate embedding for
        model_name: The Gemini model to use for embeddings

    Returns:
        List[float]: A list of floats representing the embedding vector

    Raises:
        ValueError: If the input text is empty or invalid
        Exception: For API errors after max retries

    Example:
        >>> embedding = generate_embedding("Sample movie plot")
        >>> len(embedding) == 768
        True
    """
    global LAST_API_CALL_TIME
    
    # Input validation
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
        
    text = text.strip()
    if not text:
        logging.warning("Empty text provided for embedding, returning zero vector")
        return [0.0] * 768

    # Rate limiting
    current_time = time.time()
    time_since_last_call = current_time - LAST_API_CALL_TIME
    if time_since_last_call < EMBEDDING_RATE_LIMIT_DELAY:
        time.sleep(EMBEDDING_RATE_LIMIT_DELAY - time_since_last_call)
    
    try:
        # Generate embedding with error handling for API response
        response = genai.embed_content(
            model=model_name,
            content=text,
            task_type="retrieval_document"
        )
        
        # Update last API call time
        LAST_API_CALL_TIME = time.time()
        
        # Handle different response formats
        if isinstance(response, dict):
            embedding = response.get('embedding', [])
        else:  # Handle newer SDK versions
            embedding = getattr(response, 'embedding', [])
            
        if not embedding or not isinstance(embedding, list):
            raise ValueError("Invalid embedding response format")
            
        return embedding
        
    except Exception as e:
        logging.error(f"Error generating embedding: {str(e)}")
        if "quota" in str(e).lower():
            raise RuntimeError("API quota exceeded") from e
        raise  # Re-raise to trigger retry

    return embedding


# ---------- TMDB fetchers with retries ----------
class TransientHTTPError(Exception):
    pass


def _check_response(resp: requests.Response) -> None:
    if resp.status_code >= 500:
        raise TransientHTTPError(f"TMDB server error: {resp.status_code}")
    if resp.status_code != 200:
        raise RuntimeError(f"TMDB error: {resp.status_code} - {resp.text[:200]}")


# TMDB API rate limiter (40 requests per 10 seconds)
TMDB_RATE_LIMIT = 40
TMDB_PERIOD = 10  # seconds

@sleep_and_retry
@limits(calls=TMDB_RATE_LIMIT, period=TMDB_PERIOD)
def tmdb_rate_limiter():
    """Enforce rate limiting for TMDB API."""
    pass

@retry(
    retry=retry_if_exception_type((requests.RequestException, TransientHTTPError)),
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying {retry_state.fn.__name__} after "
        f"{retry_state.outcome.exception()}... "
        f"Attempt {retry_state.attempt_number}/{settings.max_retries}"
    )
)
def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Make a GET request to TMDB API with rate limiting and retries."""
    tmdb_rate_limiter()  # Enforce rate limiting
    
    url = f"https://api.themoviedb.org/3{path}"
    headers = {"Authorization": f"Bearer {settings.tmdb_api_key}"} if settings.tmdb_api_key.startswith("eyJ") else {}
    
    try:
        if headers:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
        else:
            params = {**params, "api_key": settings.tmdb_api_key}
            resp = requests.get(url, params=params, timeout=30)
            
        _check_response(resp)
        return resp.json()
        
    except requests.RequestException as e:
        logger.error(f"TMDB API request failed: {e}")
        raise


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


def process_movie_batch(movie_batch: List[Dict[str, Any]], collection) -> Tuple[int, int]:
    """Process a batch of movies with error handling and retries."""
    inserted = 0
    skipped = 0
    
    with ThreadPoolExecutor(max_workers=settings.max_workers) as executor:
        future_to_movie = {
            executor.submit(process_single_movie, movie, collection): movie 
            for movie in movie_batch
        }
        
        for future in concurrent.futures.as_completed(future_to_movie):
            movie = future_to_movie[future]
            try:
                result = future.result()
                if result:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as exc:
                logger.error(f"Error processing movie {movie.get('id')}: {exc}", exc_info=True)
                skipped += 1
    
    return inserted, skipped

def process_single_movie(movie: Dict[str, Any], collection) -> bool:
    """Process a single movie with retries and error handling."""
    if shutdown_event:
        return False
        
    movie_id = movie.get("id")
    if not movie_id:
        return False

    try:
        details = fetch_movie_details(int(movie_id))
        if not details:
            return False
            
        movie_doc = extract_movie_document(details)
        if not movie_doc:
            return False
            
        return insert_if_new(collection, movie_doc)
        
    except Exception as exc:
        logger.error(f"Error processing movie ID {movie_id}: {exc}", exc_info=True)
        return False

def main() -> None:
    """Main entry point with batch processing and parallel execution."""
    global shutdown_event
    
    try:
        # Initialize collection with connection pooling
        collection = get_astra_collection("movies")
        
        # Track metrics
        total_inserted = 0
        total_skipped = 0
        start_time = time.time()
        
        # Process movies in batches
        logger.info("Starting movie data ingestion...")
        movie_generator = discover_movies()
        
        while True:
            if shutdown_event:
                logger.info("Shutdown requested. Stopping after current batch...")
                break
                
            # Get next batch
            batch = []
            for _ in range(settings.batch_size):
                try:
                    movie = next(movie_generator)
                    batch.append(movie)
                except StopIteration:
                    break
            
            if not batch:
                break  # No more movies to process
                
            logger.info(f"Processing batch of {len(batch)} movies...")
            
            # Process batch
            inserted, skipped = process_movie_batch(batch, collection)
            total_inserted += inserted
            total_skipped += skipped
            
            logger.info(f"Batch complete. Inserted: {inserted}, Skipped: {skipped}")
            
            # Small delay between batches to avoid rate limiting
            time.sleep(1)
            
        # Final statistics
        duration = time.time() - start_time
        logger.info(
            f"Ingestion complete. "
            f"Total inserted: {total_inserted}, "
            f"Skipped: {total_skipped}, "
            f"Duration: {duration:.2f} seconds"
        )
        
    except Exception as e:
        logger.critical(f"Fatal error in main process: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
