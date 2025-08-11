import os
import requests
from dotenv import load_dotenv
import json

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

print("🔍 Checking Your Movie Database...")
print("=" * 60)

try:
    # First, get a larger sample to see total count
    params = {'where': '{}', 'limit': 100}
    response = requests.get(base_url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        records = data.get('data', [])
        
        print(f"📊 Total records found: {len(records)}")
        print()
        
        if records:
            print("📋 Database Fields:")
            print("-" * 30)
            first_record = records[0]
            for key, value in first_record.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}... (truncated)")
                else:
                    print(f"  {key}: {value}")
            print()
            
            print("🎬 Sample Movies:")
            print("-" * 30)
            
            # Show first 10 movies
            for i, record in enumerate(records[:10], 1):
                title = record.get('title', 'No title')
                rating = record.get('rating', 'N/A')
                release_date = record.get('release_date', 'N/A')
                
                print(f"{i:2d}. {title}")
                print(f"    Rating: {rating} | Release: {release_date}")
                
                # Check if it has text content
                text = record.get('text', '')
                if text:
                    print(f"    Text: {text[:80]}...")
                print()
            
            if len(records) > 10:
                print(f"... and {len(records) - 10} more movies")
            
            print("🔍 Checking for potential issues:")
            print("-" * 30)
            
            # Check for movies with titles
            movies_with_titles = [r for r in records if r.get('title') and r.get('title').strip()]
            print(f"✅ Movies with titles: {len(movies_with_titles)}")
            
            # Check for movies with text content
            movies_with_text = [r for r in records if r.get('text') and r.get('text').strip()]
            print(f"✅ Movies with text content: {len(movies_with_text)}")
            
            # Check for movies with ratings
            movies_with_ratings = [r for r in records if r.get('rating')]
            print(f"✅ Movies with ratings: {len(movies_with_ratings)}")
            
            print()
            print("💡 If your chatbot isn't finding movies, check:")
            print("   1. Are the field names correct? (title, text, rating, etc.)")
            print("   2. Do the movies have actual text content?")
            print("   3. Is the chatbot looking for the right field names?")
            
        else:
            print("❌ No records found in database!")
            print("   Your database might be empty or the table name is wrong.")
            
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("   Check your .env file and database connection settings.")
