import os
import requests
import uuid
from dotenv import load_dotenv

load_dotenv()

print("🎬 Simple Movie Database Setup")
print("=" * 40)

# Get environment variables
db_id = os.getenv("ASTRA_DB_ID")
db_region = os.getenv("ASTRA_DB_REGION")
client_secret = os.getenv("ASTRA_CLIENT_SECRET")

if not all([db_id, db_region, client_secret]):
    print("❌ Missing environment variables!")
    print("Please check your .env file has:")
    print("  ASTRA_DB_ID=your_db_id")
    print("  ASTRA_DB_REGION=your_region")
    print("  ASTRA_CLIENT_SECRET=your_token")
    exit()

# Use the existing table structure
table_name = "ragcine"  # Your existing table
base_url = f"https://{db_id}-{db_region}.apps.astra.datastax.com/api/rest/v2/keyspaces/default_keyspace/{table_name}"

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {client_secret}',
    'X-Cassandra-Token': client_secret
}

def check_existing_data():
    """Check what data already exists in the database"""
    print("🔍 Checking existing data...")
    
    try:
        params = {'where': '{}', 'limit': 5}
        response = requests.get(base_url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            records = data.get('data', [])
            
            print(f"✅ Found {len(records)} records in existing table")
            
            if records:
                print("📊 Sample record structure:")
                sample = records[0]
                for key, value in sample.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
            
            return True
        else:
            print(f"❌ Failed to get data: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def add_movie_to_existing_table():
    """Add a movie to the existing table structure"""
    print("\n🎬 Add Movie to Existing Table")
    print("-" * 30)
    
    # Create a movie record that matches your existing structure
    movie_data = {
        "id": str(uuid.uuid4()),
        "title": "The Shawshank Redemption",
        "year": 1994,
        "genre": "Drama",
        "rating": 9.3,
        "director": "Frank Darabont",
        "cast": "Tim Robbins, Morgan Freeman",
        "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
        "runtime": 142,
        "box_office": "$58.3M",
        "streaming_platforms": "Netflix, Amazon Prime",
        "text": "The Shawshank Redemption (1994) - A powerful drama about hope and redemption starring Tim Robbins and Morgan Freeman. Directed by Frank Darabont, this film has a 9.3/10 rating and grossed $58.3M at the box office. Available on Netflix and Amazon Prime.",
        "release_date": "1994-09-22",
        "where_to_watch": "Netflix, Amazon Prime"
    }
    
    try:
        response = requests.post(base_url, headers=headers, json=movie_data)
        
        if response.status_code == 201:
            print(f"✅ Successfully added: {movie_data['title']}")
            return True
        else:
            print(f"❌ Failed to add movie: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_chatbot_connection():
    """Test if the chatbot can connect to the database"""
    print("\n🤖 Testing Chatbot Connection...")
    
    try:
        # Test the same connection the chatbot uses
        params = {'where': '{}', 'limit': 1}
        response = requests.get(base_url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            print("✅ Chatbot can connect to your database!")
            return True
        else:
            print(f"❌ Chatbot connection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def main():
    print("🚀 Starting simple database setup...")
    
    # Step 1: Check existing data
    if not check_existing_data():
        print("❌ Cannot access existing database")
        return
    
    # Step 2: Test chatbot connection
    if not test_chatbot_connection():
        print("❌ Chatbot cannot connect to database")
        return
    
    # Step 3: Try to add a sample movie
    print("\n" + "="*50)
    print("🎬 Your database is ready for the chatbot!")
    print("="*50)
    
    print("\n💡 Next steps:")
    print("1. Run your chatbot: python movie_chatbot.py")
    print("2. The chatbot will automatically connect to your existing database")
    print("3. Ask questions about movies in your database")
    
    # Ask if user wants to add a sample movie
    choice = input("\nWould you like to add a sample movie to test? (y/n): ").strip().lower()
    
    if choice == 'y':
        if add_movie_to_existing_table():
            print("\n✅ Sample movie added! Your chatbot should now be able to find it.")
        else:
            print("\n⚠️ Could not add sample movie, but your chatbot should still work with existing data.")
    
    print("\n🎉 Setup complete! Your chatbot is ready to use your existing database.")

if __name__ == "__main__":
    main()
