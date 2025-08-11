import os
import pandas as pd
from dotenv import load_dotenv
import requests

load_dotenv()

# Test Astra connection and data
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

print("🎬 Testing Your Movie Database...")
print("=" * 50)

try:
    # Get 5 records to test
    params = {'where': '{}', 'limit': 5}
    response = requests.get(base_url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        records = data.get('data', [])
        
        print(f"✅ Found {len(records)} movie records")
        print()
        
        for i, record in enumerate(records, 1):
            print(f"🎬 Movie {i}:")
            
            # Extract movie info
            title = record.get('title', 'Unknown Title')
            rating = record.get('rating', 'N/A')
            release_date = record.get('release_date', 'N/A')
            where_to_watch = record.get('where_to_watch', [])
            text_content = record.get('text', '')
            
            print(f"   Title: {title}")
            print(f"   Rating: {rating}")
            print(f"   Release Date: {release_date}")
            print(f"   Streaming: {', '.join(where_to_watch) if where_to_watch else 'Not specified'}")
            print(f"   Text: {text_content[:100]}...")
            print()
        
        print("🎯 Your chatbot should now be able to answer questions about these movies!")
        print("💡 Try asking about specific movies, ratings, or streaming platforms")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")
