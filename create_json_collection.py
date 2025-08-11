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
VECTOR_DIM = int(os.getenv("ASTRA_VECTOR_DIM", "768"))
VECTOR_METRIC = os.getenv("ASTRA_VECTOR_METRIC", "cosine")
CREATE_VECTOR_INDEX = os.getenv("ASTRA_CREATE_VECTOR_INDEX", "false").lower() in ("1", "true", "yes")

if not all([DB_ID, REGION, TOKEN]):
    print("❌ Missing ASTRA_DB_ID / ASTRA_DB_REGION / ASTRA_CLIENT_SECRET in .env")
    sys.exit(1)

collection_url = f"https://{DB_ID}-{REGION}.apps.astra.datastax.com/api/json/v1/{KEYSPACE}/{COLLECTION}"
headers = {
    "Content-Type": "application/json",
    # JSON API accepts either Authorization: Bearer or X-Cassandra-Token
    "Authorization": f"Bearer {TOKEN}",
}

print("🎯 Creating JSON API collection by inserting a placeholder document (JSON API auto-creates on first insert)")
print(f"➡️ Keyspace: {KEYSPACE}")
print(f"➡️ Collection: {COLLECTION}")

# 1) Insert a placeholder document to ensure collection exists
placeholder_doc = {
    "_id": str(uuid.uuid4()),
    "__init_marker": True,
}

try:
    resp = requests.post(collection_url, headers=headers, json={"insertOne": {"document": placeholder_doc}}, timeout=30)
    if resp.status_code in (200, 201):
        print("✅ Collection exists (placeholder inserted)")
    else:
        # If already exists, insertOne may still succeed or fail; we ignore conflicts
        print(f"⚠️ Insert placeholder response: {resp.status_code}")
        print(resp.text)
except Exception as e:
    print(f"❌ Error during insertOne: {e}")
    sys.exit(2)

# 2) Optionally create a vector index (for field 'embedding')
if CREATE_VECTOR_INDEX:
    print("🧭 Creating vector index 'embedding_idx' on field 'embedding'")
    vector_payload = {
        "createVectorIndex": {
            "definition": {
                "indexName": "embedding_idx",
                "vector": {
                    "dimension": VECTOR_DIM,
                    "metric": VECTOR_METRIC,
                    "field": "embedding"
                }
            }
        }
    }
    try:
        r = requests.post(collection_url, headers=headers, json=vector_payload, timeout=60)
        if r.status_code in (200, 201):
            print("✅ Vector index created")
        else:
            print(f"⚠️ Vector index create response: {r.status_code}")
            print(r.text)
    except Exception as e:
        print(f"❌ Error creating vector index: {e}")
        # Do not exit; collection is still usable without vector index

print("🎉 Done")
