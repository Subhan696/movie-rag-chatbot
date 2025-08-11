import os
import sys
import uuid
import requests
from dotenv import load_dotenv

load_dotenv()

DB_ID = os.getenv("ASTRA_DB_ID")
REGION = os.getenv("ASTRA_DB_REGION")
TOKEN = os.getenv("ASTRA_CLIENT_SECRET")
KEYSPACE = os.getenv("ASTRA_KEYSPACE", "default_keyspace")
COLLECTION = os.getenv("ASTRA_COLLECTION", "movies_collection")

if not all([DB_ID, REGION, TOKEN]):
    print("❌ Missing ASTRA_DB_ID / ASTRA_DB_REGION / ASTRA_CLIENT_SECRET in .env")
    sys.exit(1)

collection_url = f"https://{DB_ID}-{REGION}.apps.astra.datastax.com/api/json/v1/{KEYSPACE}/{COLLECTION}"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}",
}

# Sample movie docs
sample_movies = [
    {
        "_id": str(uuid.uuid4()),
        "title": "The Shawshank Redemption",
        "year": 1994,
        "genre": ["Drama"],
        "rating": 9.3,
        "director": "Frank Darabont",
        "cast": ["Tim Robbins", "Morgan Freeman"],
        "description": "Two imprisoned men bond over years, finding solace and eventual redemption.",
        "runtime": 142,
        "box_office": "$58.3M",
        "streaming_platforms": ["Netflix", "Amazon Prime"],
        "imdb_id": "tt0111161"
    },
    {
        "_id": str(uuid.uuid4()),
        "title": "The Dark Knight",
        "year": 2008,
        "genre": ["Action", "Crime", "Drama"],
        "rating": 9.0,
        "director": "Christopher Nolan",
        "cast": ["Christian Bale", "Heath Ledger"],
        "runtime": 152,
        "imdb_id": "tt0468569"
    }
]

print(f"🎬 Inserting {len(sample_movies)} movie documents into '{COLLECTION}' via JSON API")

try:
    payload = {"insertMany": {"documents": sample_movies}}
    resp = requests.post(collection_url, headers=headers, json=payload, timeout=60)
    if resp.status_code in (200, 201):
        print("✅ Inserted documents")
        print(resp.text)
        sys.exit(0)
    else:
        print(f"⚠️ insertMany failed ({resp.status_code}), trying insertOne loop...")
        for doc in sample_movies:
            r = requests.post(collection_url, headers=headers, json={"insertOne": {"document": doc}}, timeout=30)
            if r.status_code in (200, 201):
                print(f"  ✅ Inserted: {doc.get('title')}")
            else:
                print(f"  ❌ Failed: {doc.get('title')} ({r.status_code})")
                print(r.text)
        sys.exit(0)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(2)
