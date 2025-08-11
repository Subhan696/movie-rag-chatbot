import os
import requests
from dotenv import load_dotenv

load_dotenv()

print("🔍 Simple Database Check")
print("=" * 30)

db_id = os.getenv("ASTRA_DB_ID")
db_region = os.getenv("ASTRA_DB_REGION")
client_secret = os.getenv("ASTRA_CLIENT_SECRET")

print(f"DB ID: {db_id}")
print(f"Region: {db_region}")
print(f"Secret: {'✅ Set' if client_secret else '❌ Missing'}")

if not all([db_id, db_region, client_secret]):
    print("❌ Missing environment variables!")
    exit()

base_url = f"https://{db_id}-{db_region}.apps.astra.datastax.com/api/rest/v2/keyspaces/default_keyspace/ragcine"
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {client_secret}',
    'X-Cassandra-Token': client_secret
}

try:
    response = requests.get(base_url, headers=headers, params={'where': '{}', 'limit': 10})
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        records = data.get('data', [])
        print(f"Records found: {len(records)}")
        
        if records:
            print("First record keys:", list(records[0].keys()))
            print("Sample title:", records[0].get('title', 'No title'))
        else:
            print("No records found!")
    else:
        print(f"Error response: {response.text}")
        
except Exception as e:
    print(f"Error: {e}")
