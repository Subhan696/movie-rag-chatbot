"""
DataStax Astra Database Connector for Movie RAG Chatbot
Handles fetching movie data from Astra database using REST API
"""

import os
import json
import time
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AstraConnector:
    """Connector class for DataStax Astra database using REST API"""
    
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
        self.client_id = client_id
        self.client_secret = client_secret
        self.keyspace = keyspace
        self.table_name = table_name
        
        # Extract database ID and region from secure connect bundle path
        self.db_id, self.db_region = self._extract_db_info(secure_connect_bundle_path)
        
        # Check if we have valid configuration
        if not self.db_id or not self.db_region:
            logger.error("❌ Astra configuration incomplete")
            logger.info("📝 Please run: python test_astra_setup.py")
            self.base_url = None
            self.access_token = None
            self.headers = None
            return
        
        # Build REST API URL
        self.base_url = f"https://{self.db_id}-{self.db_region}.apps.astra.datastax.com/api/rest/v2/keyspaces/{self.keyspace}/{self.table_name}"
        
        # Get access token
        self.access_token = self._get_access_token()
        
        # Headers for API requests
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}',
            'X-Cassandra-Token': self.access_token
        }
    
    def _extract_db_info(self, bundle_path: str) -> tuple:
        """Extract database ID and region from secure connect bundle path"""
        # Get database info from environment variables
        db_id = os.getenv("ASTRA_DB_ID")
        db_region = os.getenv("ASTRA_DB_REGION")
        
        if not db_id or db_id == "your-db-id":
            logger.warning("⚠️ ASTRA_DB_ID not set or using placeholder value")
            logger.info("📝 Please set ASTRA_DB_ID in your .env file")
            return None, None
            
        if not db_region or db_region == "us-east1":
            logger.warning("⚠️ ASTRA_DB_REGION not set or using default value")
            logger.info("📝 Please set ASTRA_DB_REGION in your .env file")
            return None, None
            
        return db_id, db_region
    
    def _get_access_token(self) -> str:
        """Get access token from Astra"""
        try:
            # For now, we'll use the client secret as token
            # In production, you'd implement proper OAuth flow
            return self.client_secret
        except Exception as e:
            logger.error(f"❌ Error getting access token: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Astra database"""
        try:
            # Check if configuration is valid
            if not self.base_url or not self.headers:
                logger.error("❌ Astra configuration not set up properly")
                logger.info("📝 Please run: python test_astra_setup.py")
                return False
            
            # Use GET with query parameters (where clause required)
            params = {
                'where': '{}',  # Empty where clause as JSON string
                'limit': 1
            }
            response = requests.get(self.base_url, headers=self.headers, params=params)
            if response.status_code == 200:
                logger.info("✅ Astra connection successful")
                return True
            else:
                logger.error(f"❌ Astra connection failed: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Astra connection error: {e}")
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
            # Use GET with query parameters
            params = {
                'where': '{}',  # Empty where clause as JSON string
                'limit': limit
            }
            
            response = requests.get(self.base_url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                movies = []
                
                for row in data.get('data', []):
                    movie = {
                        'Movie Name': row.get('title', 'Unknown'),
                        'Year': row.get('year', 0),
                        'Description': row.get('description', 'No description available'),
                        'Director': row.get('director', 'Unknown'),
                        'Cast': row.get('cast', 'Unknown'),
                        'Genre': row.get('genre', 'Unknown'),
                        'Box Office': row.get('box_office', 'Unknown'),
                        'IMDb Rating': row.get('rating', 0.0),
                        'Streaming': row.get('streaming', 'Unknown'),
                        'Runtime (min)': row.get('runtime', 0),
                        'id': row.get('id'),
                        'created_at': row.get('created_at'),
                        'updated_at': row.get('updated_at')
                    }
                    movies.append(movie)
                
                logger.info(f"✅ Fetched {len(movies)} movies from Astra")
                return movies
            else:
                logger.error(f"❌ Failed to fetch movies: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return []
                
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
            # Simple search using where clause
            search_query = {
                "where": {
                    "$or": [
                        {"title": {"$contains": query}},
                        {"description": {"$contains": query}},
                        {"director": {"$contains": query}},
                        {"cast": {"$contains": query}},
                        {"genre": {"$contains": query}}
                    ]
                },
                "limit": limit
            }
            
            url = f"{self.base_url}/search"
            response = requests.post(url, headers=self.headers, json=search_query)
            
            if response.status_code == 200:
                data = response.json()
                movies = []
                
                for row in data.get('data', []):
                    movie = {
                        'Movie Name': row.get('title', 'Unknown'),
                        'Year': row.get('year', 0),
                        'Description': row.get('description', 'No description available'),
                        'Director': row.get('director', 'Unknown'),
                        'Cast': row.get('cast', 'Unknown'),
                        'Genre': row.get('genre', 'Unknown'),
                        'Box Office': row.get('box_office', 'Unknown'),
                        'IMDb Rating': row.get('rating', 0.0),
                        'Streaming': row.get('streaming', 'Unknown'),
                        'Runtime (min)': row.get('runtime', 0),
                        'id': row.get('id')
                    }
                    movies.append(movie)
                
                logger.info(f"✅ Found {len(movies)} movies matching '{query}'")
                return movies
            else:
                logger.error(f"❌ Search failed: {response.status_code}")
                return []
                
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
            url = f"{self.base_url}/{movie_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                row = data.get('data', {})
                
                movie = {
                    'Movie Name': row.get('title', 'Unknown'),
                    'Year': row.get('year', 0),
                    'Description': row.get('description', 'No description available'),
                    'Director': row.get('director', 'Unknown'),
                    'Cast': row.get('cast', 'Unknown'),
                    'Genre': row.get('genre', 'Unknown'),
                    'Box Office': row.get('box_office', 'Unknown'),
                    'IMDb Rating': row.get('rating', 0.0),
                    'Streaming': row.get('streaming', 'Unknown'),
                    'Runtime (min)': row.get('runtime', 0),
                    'id': row.get('id')
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
            search_query = {
                "where": {"genre": {"$contains": genre}},
                "orderBy": [{"column": "rating", "order": "desc"}],
                "limit": limit
            }
            
            url = f"{self.base_url}/search"
            response = requests.post(url, headers=self.headers, json=search_query)
            
            if response.status_code == 200:
                data = response.json()
                movies = []
                
                for row in data.get('data', []):
                    movie = {
                        'Movie Name': row.get('title', 'Unknown'),
                        'Year': row.get('year', 0),
                        'Description': row.get('description', 'No description available'),
                        'Director': row.get('director', 'Unknown'),
                        'Cast': row.get('cast', 'Unknown'),
                        'Genre': row.get('genre', 'Unknown'),
                        'Box Office': row.get('box_office', 'Unknown'),
                        'IMDb Rating': row.get('rating', 0.0),
                        'Streaming': row.get('streaming', 'Unknown'),
                        'Runtime (min)': row.get('runtime', 0),
                        'id': row.get('id')
                    }
                    movies.append(movie)
                
                logger.info(f"✅ Found {len(movies)} movies in genre '{genre}'")
                return movies
            else:
                logger.error(f"❌ Genre filter failed: {response.status_code}")
                return []
                
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
            search_query = {
                "orderBy": [{"column": "rating", "order": "desc"}],
                "limit": limit
            }
            
            url = f"{self.base_url}/search"
            response = requests.post(url, headers=self.headers, json=search_query)
            
            if response.status_code == 200:
                data = response.json()
                movies = []
                
                for row in data.get('data', []):
                    movie = {
                        'Movie Name': row.get('title', 'Unknown'),
                        'Year': row.get('year', 0),
                        'Description': row.get('description', 'No description available'),
                        'Director': row.get('director', 'Unknown'),
                        'Cast': row.get('cast', 'Unknown'),
                        'Genre': row.get('genre', 'Unknown'),
                        'Box Office': row.get('box_office', 'Unknown'),
                        'IMDb Rating': row.get('rating', 0.0),
                        'Streaming': row.get('streaming', 'Unknown'),
                        'Runtime (min)': row.get('runtime', 0),
                        'id': row.get('id')
                    }
                    movies.append(movie)
                
                logger.info(f"✅ Fetched {len(movies)} top-rated movies")
                return movies
            else:
                logger.error(f"❌ Top rated query failed: {response.status_code}")
                return []
                
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
            search_query = {
                "where": {"year": {"$eq": year}},
                "orderBy": [{"column": "rating", "order": "desc"}],
                "limit": limit
            }
            
            url = f"{self.base_url}/search"
            response = requests.post(url, headers=self.headers, json=search_query)
            
            if response.status_code == 200:
                data = response.json()
                movies = []
                
                for row in data.get('data', []):
                    movie = {
                        'Movie Name': row.get('title', 'Unknown'),
                        'Year': row.get('year', 0),
                        'Description': row.get('description', 'No description available'),
                        'Director': row.get('director', 'Unknown'),
                        'Cast': row.get('cast', 'Unknown'),
                        'Genre': row.get('genre', 'Unknown'),
                        'Box Office': row.get('box_office', 'Unknown'),
                        'IMDb Rating': row.get('rating', 0.0),
                        'Streaming': row.get('streaming', 'Unknown'),
                        'Runtime (min)': row.get('runtime', 0),
                        'id': row.get('id')
                    }
                    movies.append(movie)
                
                logger.info(f"✅ Found {len(movies)} movies from year {year}")
                return movies
            else:
                logger.error(f"❌ Year filter failed: {response.status_code}")
                return []
                
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
            url = f"{self.base_url}?limit=1"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                total_count = len(data.get('data', []))
                
                # For now, return basic stats
                # In a real implementation, you'd aggregate the data
                stats = {
                    'total_movies': total_count,
                    'genres': {'Unknown': total_count},  # Placeholder
                    'years': {2024: total_count},  # Placeholder
                    'last_updated': datetime.now().isoformat(),
                    'data_source': 'astra_rest_api'
                }
                
                return stats
            else:
                return {'error': f'Failed to get statistics: {response.status_code}'}
                
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return {'error': str(e)}
    
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
            
            insert_data = {
                'id': str(uuid4()),
                'title': movie_data.get('title', ''),
                'year': movie_data.get('year', 0),
                'description': movie_data.get('description', ''),
                'director': movie_data.get('director', ''),
                'cast': movie_data.get('cast', ''),
                'genre': movie_data.get('genre', ''),
                'box_office': movie_data.get('box_office', ''),
                'rating': movie_data.get('rating', 0.0),
                'streaming': movie_data.get('streaming', ''),
                'runtime': movie_data.get('runtime', 0),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=insert_data)
            
            if response.status_code == 201:
                logger.info(f"✅ Inserted movie: {movie_data.get('title', 'Unknown')}")
                return True
            else:
                logger.error(f"❌ Failed to insert movie: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error inserting movie: {e}")
            return False
    
    def close(self):
        """Close the database connection (no-op for REST API)"""
        logger.info("✅ Astra REST API connection closed")

# Example usage and testing
if __name__ == "__main__":
    # Test the connector with environment variables
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Get configuration from environment
    db_id = os.getenv("ASTRA_DB_ID")
    db_region = os.getenv("ASTRA_DB_REGION")
    client_id = os.getenv("ASTRA_CLIENT_ID")
    client_secret = os.getenv("ASTRA_CLIENT_SECRET")
    keyspace = os.getenv("ASTRA_KEYSPACE", "default_keyspace")
    table_name = os.getenv("ASTRA_TABLE_NAME", "ragcine")
    
    if not all([db_id, db_region, client_id, client_secret]):
        print("❌ Missing Astra configuration")
        print("📝 Please set up your .env file with Astra credentials")
        print("📝 Run: python test_astra_setup.py")
        exit(1)
    
    # Test the connector
    connector = AstraConnector(
        secure_connect_bundle_path="dummy",  # Not used in REST API
        client_id=client_id,
        client_secret=client_secret,
        keyspace=keyspace,
        table_name=table_name
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
