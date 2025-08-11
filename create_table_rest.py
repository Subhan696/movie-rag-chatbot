import os
import sys
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

schema_url = f"https://{DB_ID}-{REGION}.apps.astra.datastax.com/api/rest/v2/schemas/keyspaces/{KEYSPACE}/tables"
headers = {
    "Content-Type": "application/json",
    "X-Cassandra-Token": TOKEN,
}

print("🧱 Creating Cassandra table via REST schema API")
print(f"➡️ Keyspace: {KEYSPACE}")
print(f"➡️ Table: {TABLE}")

payload = {
    "name": TABLE,
    "ifNotExists": True,
    "columnDefinitions": [
        {"name": "id", "typeDefinition": "uuid", "static": False},
        {"name": "title", "typeDefinition": "text", "static": False},
        {"name": "year", "typeDefinition": "int", "static": False},
        {"name": "genre", "typeDefinition": "text", "static": False},
        {"name": "rating", "typeDefinition": "float", "static": False},
        {"name": "director", "typeDefinition": "text", "static": False},
        {"name": "cast", "typeDefinition": "text", "static": False},
        {"name": "description", "typeDefinition": "text", "static": False},
        {"name": "runtime", "typeDefinition": "int", "static": False},
        {"name": "box_office", "typeDefinition": "text", "static": False},
        {"name": "streaming_platforms", "typeDefinition": "text", "static": False},
        {"name": "imdb_id", "typeDefinition": "text", "static": False}
    ],
    "primaryKey": {
        "partitionKey": ["id"]
    }
}

try:
    resp = requests.post(schema_url, headers=headers, json=payload, timeout=30)
    if resp.status_code in (200, 201, 409):
        print(f"✅ Table ready (status {resp.status_code})")
        print(resp.text)
        sys.exit(0)
    else:
        print(f"❌ Failed to create table: {resp.status_code}")
        print(resp.text)
        sys.exit(2)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(3)
