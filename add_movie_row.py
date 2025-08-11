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
TABLE = os.getenv("ASTRA_TABLE_NAME", "movies")

if not all([DB_ID, REGION, TOKEN]):
    print("❌ Missing ASTRA_DB_ID / ASTRA_DB_REGION / ASTRA_CLIENT_SECRET in .env")
    sys.exit(1)

rows_url = f"https://{DB_ID}-{REGION}.apps.astra.datastax.com/api/rest/v2/keyspaces/{KEYSPACE}/{TABLE}"
headers = {
    "Content-Type": "application/json",
    "X-Cassandra-Token": TOKEN,
}

rows = [
    {
        "id": str(uuid.uuid4()),
        "title": "Inception",
        "year": 2010,
        "genre": "Sci-Fi",
        "rating": 8.8,
        "director": "Christopher Nolan",
        "cast": "Leonardo DiCaprio, Joseph Gordon-Levitt",
        "description": "A thief who steals corporate secrets through dream-sharing technology is tasked with planting an idea.",
        "runtime": 148,
        "box_office": "$836.8M",
        "streaming_platforms": "Netflix",
        "imdb_id": "tt1375666"
    }
]

print(f"🎬 Inserting {len(rows)} row(s) into {KEYSPACE}.{TABLE}")

try:
    resp = requests.post(rows_url, headers=headers, json=rows, timeout=30)
    if resp.status_code in (200, 201):
        print("✅ Inserted row(s)")
        print(resp.text)
        sys.exit(0)
    else:
        print(f"❌ Failed to insert: {resp.status_code}")
        print(resp.text)
        sys.exit(2)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(3)
