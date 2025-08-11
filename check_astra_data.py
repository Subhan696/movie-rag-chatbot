"""
Check what data is actually in your Astra database
"""

import os
import requests
from dotenv import load_dotenv
import json

def check_astra_data():
    """Check what data is in your Astra database"""
    print("🔍 Checking Your Astra Database Data...")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
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
    
    # Get a few records to see what we have
    params = {'where': '{}', 'limit': 3}
    
    try:
        response = requests.get(base_url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            records = data.get('data', [])
            
            print(f"✅ Successfully retrieved {len(records)} records")
            print()
            
            if records:
                print("📋 DATA STRUCTURE ANALYSIS:")
                print("=" * 50)
                
                # Show first record structure
                first_record = records[0]
                print(f"📊 Record has {len(first_record)} fields:")
                print()
                
                for key, value in first_record.items():
                    if isinstance(value, str):
                        if len(value) > 200:
                            print(f"  {key}: {type(value).__name__} = {value[:200]}... (truncated)")
                        else:
                            print(f"  {key}: {type(value).__name__} = {value}")
                    else:
                        print(f"  {key}: {type(value).__name__} = {value}")
                
                print()
                print("🔍 TEXT FIELD ANALYSIS:")
                print("=" * 50)
                
                # Check for text fields
                text_fields = []
                vector_fields = []
                
                for key, value in first_record.items():
                    if isinstance(value, str):
                        if ',' in value and len(value) > 100:
                            # Likely a vector
                            vector_fields.append(key)
                            print(f"  🧮 {key}: Vector field ({len(value.split(','))} dimensions)")
                        elif len(value) < 1000 and not value.isdigit():
                            # Likely text
                            text_fields.append(key)
                            print(f"  📝 {key}: Text field = '{value}'")
                        else:
                            print(f"  ❓ {key}: Unknown string field = '{value[:100]}...'")
                
                print()
                print("💡 RECOMMENDATIONS:")
                print("=" * 50)
                
                if text_fields:
                    print("✅ You have text fields that can be used for movie data!")
                    print(f"   Text fields found: {text_fields}")
                    print("   Your chatbot should work with these fields.")
                else:
                    print("❌ No text fields found for movie data!")
                    print("   Your database contains only vector embeddings.")
                    print("   You need to add movie text data to make the chatbot work.")
                
                if vector_fields:
                    print(f"🧮 You have {len(vector_fields)} vector fields for semantic search")
                    print("   These can be used for advanced similarity matching.")
                
                print()
                print("🚀 NEXT STEPS:")
                print("-" * 30)
                
                if not text_fields:
                    print("1. Add movie text data to your Astra database")
                    print("2. Include fields like: title, description, genre, director, cast")
                    print("3. Then your chatbot will be able to provide meaningful responses")
                else:
                    print("1. Your chatbot should work with the existing text fields")
                    print("2. Try running: python movie_chatbot.py")
                    print("3. Ask questions about movies to test it")
                
                return True
            else:
                print("❌ No data found in your table")
                print("💡 You need to add data to your Astra database first")
                return False
        else:
            print(f"❌ Failed to retrieve data: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_astra_data()
