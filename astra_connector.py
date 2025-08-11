"""
DataStax Astra Database Connector for Movie RAG Chatbot
Handles fetching movie data from Astra database
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AstraConnector:
    """Connector class for DataStax Astra database"""
    
    def __init__(self, 
                 secure_connect_bundle_path: str,
                 client_id: str,
                 client_secret: str,
                 keyspace: str = "movies",
                 table_name: str = "movies"):
        """
        Initialize Astra connector
        
        Args:
            secure_connect_bundle_path: Path to Astra secure connect bundle
            client_id: Astra client ID
            client_secret: Astra client secret
            keyspace: Database keyspace name
            table_name: Table name containing movie data
        """
        self.secure_connect_bundle_path = secure_connect_bundle_path
        self.client_id = client_id
        self.client_secret = client_secret
        self.keyspace = keyspace
        self.table_name = table_name
        self.session = None
        self.cluster = None
        
        # Initialize connection
        self._connect()
    
    def _connect(self):
        """Establish connection to Astra database"""
        try:
            # Create auth provider
            auth_provider = PlainTextAuthProvider(
                username=self.client_id,
                password=self.client_secret
            )
            
            # Create cluster
            self.cluster = Cluster(
                cloud={'secure_connect_bundle': self.secure_connect_bundle_path},
                auth_provider=auth_provider
            )
            
            # Create session
            self.session = self.cluster.connect()
            
            # Set keyspace
            self.session.set_keyspace(self.keyspace)
            
            logger.info("✅ Astra connection successful")
            
        except Exception as e:
            logger.error(f"❌ Astra connection failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Astra database"""
        try:
            if not self.session:
                return False
            
            # Simple query to test connection
            query = f"SELECT COUNT(*) FROM {self.table_name} LIMIT 1"
            result = self.session.execute(query)
            
            logger.info("✅ Astra connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Astra connection test failed: {e}")
            return False
    
    def get_all_movies(self, limit: int = 1000) -> List[Dict]:
        """
        Fetch all movies from Astra
        
        Args:
            limit: Maximum number of records to fetch
        
        Returns:
            List of movie dictionaries
        """
        try:
            query = f"SELECT * FROM {self.table_name} LIMIT {limit}"
            result = self.session.execute(query)
            
            movies = []
            for row in result:
                movie = {
                    'Movie Name': getattr(row, 'title', 'Unknown'),
                    'Year': getattr(row, 'year', 0),
                    'Description': getattr(row, 'description', 'No description available'),
                    'Director': getattr(row, 'director', 'Unknown'),
                    'Cast': getattr(row, 'cast', 'Unknown'),
                    'Genre': getattr(row, 'genre', 'Unknown'),
                    'Box Office': getattr(row, 'box_office', 'Unknown'),
                    'IMDb Rating': getattr(row, 'rating', 0.0),
                    'Streaming': getattr(row, 'streaming', 'Unknown'),
                    'Runtime (min)': getattr(row, 'runtime', 0),
                    'id': getattr(row, 'id', None),
                    'created_at': getattr(row, 'created_at', None),
                    'updated_at': getattr(row, 'updated_at', None)
                }
                movies.append(movie)
            
            logger.info(f"✅ Fetched {len(movies)} movies from Astra")
            return movies
            
        except Exception as e:
            logger.error(f"❌ Error fetching movies from Astra: {e}")
            return []
    
    def search_movies(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search movies in Astra using text search
        
        Args:
            query: Search query
            limit: Maximum number of results
        
        Returns:
            List of matching movies
        """
        try:
            # Using LIKE for text search (you might want to use Solr/Elasticsearch for better search)
            search_query = f"""
                SELECT * FROM {self.table_name} 
                WHERE title CONTAINS %s 
                   OR description CONTAINS %s 
                   OR director CONTAINS %s 
                   OR cast CONTAINS %s 
                   OR genre CONTAINS %s 
                LIMIT {limit}
            """
            
            result = self.session.execute(search_query, (query, query, query, query, query))
            
            movies = []
            for row in result:
                movie = {
                    'Movie Name': getattr(row, 'title', 'Unknown'),
                    'Year': getattr(row, 'year', 0),
                    'Description': getattr(row, 'description', 'No description available'),
                    'Director': getattr(row, 'director', 'Unknown'),
                    'Cast': getattr(row, 'cast', 'Unknown'),
                    'Genre': getattr(row, 'genre', 'Unknown'),
                    'Box Office': getattr(row, 'box_office', 'Unknown'),
                    'IMDb Rating': getattr(row, 'rating', 0.0),
                    'Streaming': getattr(row, 'streaming', 'Unknown'),
                    'Runtime (min)': getattr(row, 'runtime', 0),
                    'id': getattr(row, 'id', None)
                }
                movies.append(movie)
            
            logger.info(f"✅ Found {len(movies)} movies matching '{query}'")
            return movies
            
        except Exception as e:
            logger.error(f"❌ Error searching movies: {e}")
            return []
    
    def get_movie_by_id(self, movie_id: str) -> Optional[Dict]:
        """
        Get a specific movie by ID
        
        Args:
            movie_id: Movie ID
        
        Returns:
            Movie data or None if not found
        """
        try:
            query = f"SELECT * FROM {self.table_name} WHERE id = %s"
            result = self.session.execute(query, (movie_id,))
            
            row = result.one()
            if row:
                movie = {
                    'Movie Name': getattr(row, 'title', 'Unknown'),
                    'Year': getattr(row, 'year', 0),
                    'Description': getattr(row, 'description', 'No description available'),
                    'Director': getattr(row, 'director', 'Unknown'),
                    'Cast': getattr(row, 'cast', 'Unknown'),
                    'Genre': getattr(row, 'genre', 'Unknown'),
                    'Box Office': getattr(row, 'box_office', 'Unknown'),
                    'IMDb Rating': getattr(row, 'rating', 0.0),
                    'Streaming': getattr(row, 'streaming', 'Unknown'),
                    'Runtime (min)': getattr(row, 'runtime', 0),
                    'id': getattr(row, 'id', None)
                }
                return movie
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error fetching movie {movie_id}: {e}")
            return None
    
    def get_movies_by_genre(self, genre: str, limit: int = 20) -> List[Dict]:
        """
        Get movies by genre
        
        Args:
            genre: Genre to filter by
            limit: Maximum number of results
        
        Returns:
            List of movies in the specified genre
        """
        try:
            query = f"""
                SELECT * FROM {self.table_name} 
                WHERE genre CONTAINS %s 
                ORDER BY rating DESC 
                LIMIT {limit}
            """
            
            result = self.session.execute(query, (genre,))
            
            movies = []
            for row in result:
                movie = {
                    'Movie Name': getattr(row, 'title', 'Unknown'),
                    'Year': getattr(row, 'year', 0),
                    'Description': getattr(row, 'description', 'No description available'),
                    'Director': getattr(row, 'director', 'Unknown'),
                    'Cast': getattr(row, 'cast', 'Unknown'),
                    'Genre': getattr(row, 'genre', 'Unknown'),
                    'Box Office': getattr(row, 'box_office', 'Unknown'),
                    'IMDb Rating': getattr(row, 'rating', 0.0),
                    'Streaming': getattr(row, 'streaming', 'Unknown'),
                    'Runtime (min)': getattr(row, 'runtime', 0),
                    'id': getattr(row, 'id', None)
                }
                movies.append(movie)
            
            logger.info(f"✅ Found {len(movies)} movies in genre '{genre}'")
            return movies
            
        except Exception as e:
            logger.error(f"❌ Error filtering by genre: {e}")
            return []
    
    def get_top_rated_movies(self, limit: int = 10) -> List[Dict]:
        """
        Get top-rated movies
        
        Args:
            limit: Maximum number of results
        
        Returns:
            List of top-rated movies
        """
        try:
            query = f"""
                SELECT * FROM {self.table_name} 
                ORDER BY rating DESC 
                LIMIT {limit}
            """
            
            result = self.session.execute(query)
            
            movies = []
            for row in result:
                movie = {
                    'Movie Name': getattr(row, 'title', 'Unknown'),
                    'Year': getattr(row, 'year', 0),
                    'Description': getattr(row, 'description', 'No description available'),
                    'Director': getattr(row, 'director', 'Unknown'),
                    'Cast': getattr(row, 'cast', 'Unknown'),
                    'Genre': getattr(row, 'genre', 'Unknown'),
                    'Box Office': getattr(row, 'box_office', 'Unknown'),
                    'IMDb Rating': getattr(row, 'rating', 0.0),
                    'Streaming': getattr(row, 'streaming', 'Unknown'),
                    'Runtime (min)': getattr(row, 'runtime', 0),
                    'id': getattr(row, 'id', None)
                }
                movies.append(movie)
            
            logger.info(f"✅ Fetched {len(movies)} top-rated movies")
            return movies
            
        except Exception as e:
            logger.error(f"❌ Error fetching top rated movies: {e}")
            return []
    
    def get_movies_by_year(self, year: int, limit: int = 20) -> List[Dict]:
        """
        Get movies by year
        
        Args:
            year: Year to filter by
            limit: Maximum number of results
        
        Returns:
            List of movies from the specified year
        """
        try:
            query = f"""
                SELECT * FROM {self.table_name} 
                WHERE year = %s 
                ORDER BY rating DESC 
                LIMIT {limit}
            """
            
            result = self.session.execute(query, (year,))
            
            movies = []
            for row in result:
                movie = {
                    'Movie Name': getattr(row, 'title', 'Unknown'),
                    'Year': getattr(row, 'year', 0),
                    'Description': getattr(row, 'description', 'No description available'),
                    'Director': getattr(row, 'director', 'Unknown'),
                    'Cast': getattr(row, 'cast', 'Unknown'),
                    'Genre': getattr(row, 'genre', 'Unknown'),
                    'Box Office': getattr(row, 'box_office', 'Unknown'),
                    'IMDb Rating': getattr(row, 'rating', 0.0),
                    'Streaming': getattr(row, 'streaming', 'Unknown'),
                    'Runtime (min)': getattr(row, 'runtime', 0),
                    'id': getattr(row, 'id', None)
                }
                movies.append(movie)
            
            logger.info(f"✅ Found {len(movies)} movies from year {year}")
            return movies
            
        except Exception as e:
            logger.error(f"❌ Error filtering by year: {e}")
            return []
    
    def get_movie_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the movie database
        
        Returns:
            Dictionary with database statistics
        """
        try:
            # Get total count
            count_query = f"SELECT COUNT(*) as total FROM {self.table_name}"
            count_result = self.session.execute(count_query)
            total_count = count_result.one().total
            
            # Get genre distribution
            genre_query = f"SELECT genre, COUNT(*) as count FROM {self.table_name} GROUP BY genre"
            genre_result = self.session.execute(genre_query)
            
            genres = {}
            for row in genre_result:
                genres[getattr(row, 'genre', 'Unknown')] = getattr(row, 'count', 0)
            
            # Get year distribution
            year_query = f"SELECT year, COUNT(*) as count FROM {self.table_name} GROUP BY year ORDER BY year DESC"
            year_result = self.session.execute(year_query)
            
            years = {}
            for row in year_result:
                years[getattr(row, 'year', 0)] = getattr(row, 'count', 0)
            
            return {
                'total_movies': total_count,
                'genres': genres,
                'years': years,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return {'error': str(e)}
    
    def create_movie_table(self):
        """
        Create the movies table if it doesn't exist
        This is a helper method for setting up the database schema
        """
        try:
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {self.keyspace}.{self.table_name} (
                    id uuid PRIMARY KEY,
                    title text,
                    year int,
                    description text,
                    director text,
                    cast text,
                    genre text,
                    box_office text,
                    rating decimal,
                    streaming text,
                    runtime int,
                    created_at timestamp,
                    updated_at timestamp
                )
            """
            
            self.session.execute(create_table_query)
            logger.info(f"✅ Created table {self.table_name}")
            
        except Exception as e:
            logger.error(f"❌ Error creating table: {e}")
    
    def insert_movie(self, movie_data: Dict) -> bool:
        """
        Insert a new movie into the database
        
        Args:
            movie_data: Movie data dictionary
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from uuid import uuid4
            from datetime import datetime
            
            insert_query = f"""
                INSERT INTO {self.table_name} (
                    id, title, year, description, director, cast, 
                    genre, box_office, rating, streaming, runtime, 
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            now = datetime.now()
            
            self.session.execute(insert_query, (
                uuid4(),
                movie_data.get('title', ''),
                movie_data.get('year', 0),
                movie_data.get('description', ''),
                movie_data.get('director', ''),
                movie_data.get('cast', ''),
                movie_data.get('genre', ''),
                movie_data.get('box_office', ''),
                movie_data.get('rating', 0.0),
                movie_data.get('streaming', ''),
                movie_data.get('runtime', 0),
                now,
                now
            ))
            
            logger.info(f"✅ Inserted movie: {movie_data.get('title', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inserting movie: {e}")
            return False
    
    def close(self):
        """Close the database connection"""
        try:
            if self.session:
                self.session.shutdown()
            if self.cluster:
                self.cluster.shutdown()
            logger.info("✅ Astra connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing connection: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Test the connector
    connector = AstraConnector(
        secure_connect_bundle_path="path/to/secure-connect-database.zip",
        client_id="your_client_id",
        client_secret="your_client_secret",
        keyspace="movies",
        table_name="movies"
    )
    
    # Test connection
    if connector.test_connection():
        # Get all movies
        movies = connector.get_all_movies(limit=10)
        print(f"Found {len(movies)} movies")
        
        # Get statistics
        stats = connector.get_movie_statistics()
        print(f"Database statistics: {stats}")
        
        # Search for movies
        search_results = connector.search_movies("action", limit=5)
        print(f"Search results: {len(search_results)} movies")
        
        # Close connection
        connector.close()
    else:
        print("Failed to connect to Astra")
