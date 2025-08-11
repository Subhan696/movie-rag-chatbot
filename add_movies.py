import os
import requests
import uuid
from dotenv import load_dotenv

load_dotenv()

print("🎬 Movie Database Manager")
print("=" * 40)

# Get environment variables
db_id = os.getenv("ASTRA_DB_ID")
db_region = os.getenv("ASTRA_DB_REGION")
client_secret = os.getenv("ASTRA_CLIENT_SECRET")
table_name = os.getenv("ASTRA_TABLE_NAME", "movies")

if not all([db_id, db_region, client_secret]):
    print("❌ Missing environment variables!")
    exit()

base_url = f"https://{db_id}-{db_region}.apps.astra.datastax.com/api/rest/v2/keyspaces/default_keyspace/{table_name}"
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {client_secret}',
    'X-Cassandra-Token': client_secret
}

def add_movie():
    """Add a single movie manually"""
    print("\n🎬 Add New Movie")
    print("-" * 20)
    
    movie = {
        "id": str(uuid.uuid4()),
        "title": input("Title: ").strip(),
        "year": int(input("Year: ").strip()),
        "genre": input("Genre: ").strip(),
        "rating": float(input("Rating (0-10): ").strip()),
        "director": input("Director: ").strip(),
        "cast": input("Cast: ").strip(),
        "description": input("Description: ").strip(),
        "runtime": int(input("Runtime (minutes): ").strip()),
        "box_office": input("Box Office: ").strip(),
        "streaming_platforms": input("Streaming Platforms: ").strip(),
        "poster_url": input("Poster URL (optional): ").strip() or "https://example.com/default.jpg",
        "imdb_id": input("IMDb ID (optional): ").strip() or ""
    }
    
    try:
        response = requests.post(base_url, headers=headers, json=movie)
        if response.status_code == 201:
            print(f"✅ Successfully added: {movie['title']}")
        else:
            print(f"❌ Failed to add movie: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

def add_popular_movies():
    """Add a list of popular movies"""
    popular_movies = [
        {
            "title": "The Dark Knight",
            "year": 2008,
            "genre": "Action, Crime, Drama",
            "rating": 9.0,
            "director": "Christopher Nolan",
            "cast": "Christian Bale, Heath Ledger, Aaron Eckhart",
            "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
            "runtime": 152,
            "box_office": "$1.005B",
            "streaming_platforms": "HBO Max, Amazon Prime",
            "imdb_id": "tt0468569"
        },
        {
            "title": "Fight Club",
            "year": 1999,
            "genre": "Drama",
            "rating": 8.8,
            "director": "David Fincher",
            "cast": "Brad Pitt, Edward Norton, Helena Bonham Carter",
            "description": "An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.",
            "runtime": 139,
            "box_office": "$101.2M",
            "streaming_platforms": "Hulu, Amazon Prime",
            "imdb_id": "tt0133093"
        },
        {
            "title": "Inception",
            "year": 2010,
            "genre": "Action, Adventure, Sci-Fi",
            "rating": 8.8,
            "director": "Christopher Nolan",
            "cast": "Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page",
            "description": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
            "runtime": 148,
            "box_office": "$836.8M",
            "streaming_platforms": "Netflix, Amazon Prime",
            "imdb_id": "tt1375666"
        },
        {
            "title": "The Matrix",
            "year": 1999,
            "genre": "Action, Sci-Fi",
            "rating": 8.7,
            "director": "Lana Wachowski, Lilly Wachowski",
            "cast": "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss",
            "description": "A computer programmer discovers that reality as he knows it is a simulation created by machines, and joins a rebellion to break free.",
            "runtime": 136,
            "box_office": "$463.5M",
            "streaming_platforms": "HBO Max, Amazon Prime",
            "imdb_id": "tt0133093"
        },
        {
            "title": "Goodfellas",
            "year": 1990,
            "genre": "Biography, Crime, Drama",
            "rating": 8.7,
            "director": "Martin Scorsese",
            "cast": "Robert De Niro, Ray Liotta, Joe Pesci",
            "description": "The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners Jimmy Conway and Tommy DeVito.",
            "runtime": 146,
            "box_office": "$46.8M",
            "streaming_platforms": "HBO Max, Amazon Prime",
            "imdb_id": "tt0099685"
        }
    ]
    
    print(f"\n🎬 Adding {len(popular_movies)} popular movies...")
    
    for movie_data in popular_movies:
        movie = {
            "id": str(uuid.uuid4()),
            "poster_url": "https://example.com/default.jpg",
            **movie_data
        }
        
        try:
            response = requests.post(base_url, headers=headers, json=movie)
            if response.status_code == 201:
                print(f"✅ Added: {movie['title']} ({movie['year']})")
            else:
                print(f"❌ Failed to add {movie['title']}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error adding {movie['title']}: {e}")

def view_movies():
    """View all movies in the database"""
    try:
        params = {'where': '{}', 'limit': 20}
        response = requests.get(base_url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            movies = data.get('data', [])
            
            print(f"\n📋 Found {len(movies)} movies:")
            print("-" * 50)
            
            for movie in movies:
                print(f"🎬 {movie.get('title', 'Unknown')} ({movie.get('year', 'N/A')})")
                print(f"   Rating: {movie.get('rating', 'N/A')}/10 | Genre: {movie.get('genre', 'N/A')}")
                print(f"   Director: {movie.get('director', 'N/A')}")
                print()
        else:
            print(f"❌ Failed to get movies: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    while True:
        print("\n🎬 Movie Database Manager")
        print("1. Add a single movie")
        print("2. Add popular movies (batch)")
        print("3. View all movies")
        print("4. Exit")
        
        choice = input("\nChoose an option (1-4): ").strip()
        
        if choice == "1":
            add_movie()
        elif choice == "2":
            add_popular_movies()
        elif choice == "3":
            view_movies()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
