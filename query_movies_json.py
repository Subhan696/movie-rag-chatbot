import os
import sys
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

print("🎯 Query JSON API collection")

# Count
count_payload = {"estimatedDocumentCount": {}}
resp = requests.post(collection_url, headers=headers, json=count_payload, timeout=30)
print("Count status:", resp.status_code)
print(resp.text)

# Find first 3
find_payload = {"find": {"options": {"limit": 3}}}
resp2 = requests.post(collection_url, headers=headers, json=find_payload, timeout=30)
print("Find status:", resp2.status_code)
print(resp2.text)
