import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

print("🎬 Creating New Movie Database Table...")
print("=" * 50)

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

# Base URL for Astra REST API
base_url = f"https://{db_id}-{db_region}.apps.astra.datastax.com/api/rest/v2/keyspaces/default_keyspace"

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {client_secret}',
    'X-Cassandra-Token': client_secret
}

def create_movie_table():
    """Create a new movie table with proper structure"""
    
    # Table name for the new movie database
    table_name = "movies"
    
    # Define the table schema
    table_schema = {
        "name": table_name,
        "columnDefinitions": [
            {
                "name": "id",
                "typeDefinition": "uuid"
            },
            {
                "name": "title",
                "typeDefinition": "text"
            },
            {
                "name": "year",
                "typeDefinition": "int"
            },
            {
                "name": "genre",
                "typeDefinition": "text"
            },
            {
                "name": "rating",
                "typeDefinition": "float"
            },
            {
                "name": "director",
                "typeDefinition": "text"
            },
            {
                "name": "cast",
                "typeDefinition": "text"
            },
            {
                "name": "description",
                "typeDefinition": "text"
            },
            {
                "name": "runtime",
                "typeDefinition": "int"
            },
            {
                "name": "box_office",
                "typeDefinition": "text"
            },
            {
                "name": "streaming_platforms",
                "typeDefinition": "text"
            },
            {
                "name": "poster_url",
                "typeDefinition": "text"
            },
            {
                "name": "imdb_id",
                "typeDefinition": "text"
            }
        ],
        "primaryKey": {
            "partitionKey": ["id"]
        }
    }
    
    print(f"📋 Creating table: {table_name}")
    print("📊 Table structure:")
    for col in table_schema["columnDefinitions"]:
        print(f"  • {col['name']}: {col['typeDefinition']}")
    
    try:
        # Create the table
        create_url = f"{base_url}/tables"
        response = requests.post(create_url, headers=headers, json=table_schema)
        
        if response.status_code == 201:
            print(f"✅ Successfully created table: {table_name}")
            return table_name
        else:
            print(f"❌ Failed to create table: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        return None

def add_sample_movies(table_name):
    """Add some sample movies to the new table"""
    
    sample_movies = [
        {
            "id": "550e8400-e29b-41d4-a716-446655440001",
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
            "poster_url": "https://example.com/shawshank.jpg",
            "imdb_id": "tt0111161"
        },
        {
            "id": "550e8400-e29b-41d4-a716-446655440002",
            "title": "The Godfather",
            "year": 1972,
            "genre": "Crime, Drama",
            "rating": 9.2,
            "director": "Francis Ford Coppola",
            "cast": "Marlon Brando, Al Pacino",
            "description": "The aging patriarch of an organized crime dynasty transfers control to his reluctant son.",
            "runtime": 175,
            "box_office": "$245M",
            "streaming_platforms": "Amazon Prime, Paramount+",
            "poster_url": "https://example.com/godfather.jpg",
            "imdb_id": "tt0068646"
        },
        {
            "id": "550e8400-e29b-41d4-a716-446655440003",
            "title": "Pulp Fiction",
            "year": 1994,
            "genre": "Crime, Drama",
            "rating": 8.9,
            "director": "Quentin Tarantino",
            "cast": "John Travolta, Samuel L. Jackson",
            "description": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.",
            "runtime": 154,
            "box_office": "$213.9M",
            "streaming_platforms": "Hulu, Amazon Prime",
            "poster_url": "https://example.com/pulp-fiction.jpg",
            "imdb_id": "tt0110912"
        }
    ]
    
    print(f"\n🎬 Adding {len(sample_movies)} sample movies...")
    
    for movie in sample_movies:
        try:
            add_url = f"{base_url}/{table_name}"
            response = requests.post(add_url, headers=headers, json=movie)
            
            if response.status_code == 201:
                print(f"✅ Added: {movie['title']} ({movie['year']})")
            else:
                print(f"❌ Failed to add {movie['title']}: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error adding {movie['title']}: {e}")

def main():
    print("🚀 Starting movie database setup...")
    
    # Create the new table
    table_name = create_movie_table()
    
    if table_name:
        print(f"\n📝 Table '{table_name}' created successfully!")
        
        # Add sample movies
        add_sample_movies(table_name)
        
        print(f"\n🎉 Setup complete!")
        print(f"📋 Your new table: {table_name}")
        print(f"🔗 You can now update your .env file:")
        print(f"   ASTRA_TABLE_NAME={table_name}")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Update your .env file with ASTRA_TABLE_NAME={table_name}")
        print(f"   2. Add more movies to the database")
        print(f"   3. Run your chatbot to test the new database")
        
    else:
        print("❌ Failed to create table. Please check your Astra configuration.")

if __name__ == "__main__":
    main()
