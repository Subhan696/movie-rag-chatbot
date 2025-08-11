import os
import requests
from dotenv import load_dotenv

load_dotenv()

db_id = os.getenv("ASTRA_DB_ID")
db_region = os.getenv("ASTRA_DB_REGION")
client_secret = os.getenv("ASTRA_CLIENT_SECRET")
keyspace = os.getenv("ASTRA_KEYSPACE", "default_keyspace")
table_name = os.getenv("ASTRA_TABLE_NAME", "ragcine")

base_url = f"https://{db_id}-{db_region}.apps.astra.datastax.com/api/rest/v2/keyspaces/{keyspace}/{table_name}"
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {client_secret}',
    'X-Cassandra-Token': client_secret
}

params = {'where': '{}', 'limit': 2}

try:
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        records = data.get('data', [])
        print(f"Found {len(records)} records")
        if records:
            print("First record fields:")
            for key, value in records[0].items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
