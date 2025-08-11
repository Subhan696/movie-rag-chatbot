"""
Test script to explore the data structure in Astra database
"""

import os
import requests
from dotenv import load_dotenv

def explore_astra_data():
    """Explore the data structure in Astra database"""
    print("🔍 Exploring Astra Database Data Structure...")
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
    
    # Get a few records to see the data structure
    params = {
        'where': '{}',
        'limit': 3
    }
    
    try:
        response = requests.get(base_url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            movies = data.get('data', [])
            
            print(f"✅ Successfully retrieved {len(movies)} records")
            print()
            
            if movies:
                print("📋 Sample data structure:")
                print("-" * 40)
                
                # Show the first record structure
                first_movie = movies[0]
                print("First record fields:")
                for key, value in first_movie.items():
                    print(f"  {key}: {type(value).__name__} = {value}")
                
                print()
                print("📊 Sample records:")
                print("-" * 40)
                
                for i, movie in enumerate(movies, 1):
                    print(f"\nRecord {i}:")
                    for key, value in movie.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {value[:100]}...")
                        else:
                            print(f"  {key}: {value}")
                
                # Check if we have the expected movie fields
                expected_fields = ['title', 'year', 'description', 'director', 'cast', 'genre', 'rating']
                available_fields = list(first_movie.keys())
                
                print()
                print("🔍 Field mapping analysis:")
                print("-" * 40)
                
                for field in expected_fields:
                    if field in available_fields:
                        print(f"✅ {field}: {available_fields[available_fields.index(field)]}")
                    else:
                        # Try to find similar fields
                        similar = [f for f in available_fields if field in f.lower()]
                        if similar:
                            print(f"⚠️  {field}: Similar fields found - {similar}")
                        else:
                            print(f"❌ {field}: Not found")
                
                return True
            else:
                print("❌ No data found in the table")
                return False
        else:
            print(f"❌ Failed to retrieve data: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error exploring data: {e}")
        return False

if __name__ == "__main__":
    explore_astra_data()
