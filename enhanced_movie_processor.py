"""
Enhanced Movie Data Processor
Supports multiple data sources: Astra, Excel, Strapi, and fallback sample data
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import logging
import os
from datetime import datetime

# Import connectors
try:
    from astra_connector import AstraConnector
except ImportError:
    AstraConnector = None

try:
    from strapi_connector import StrapiConnector
except ImportError:
    StrapiConnector = None

from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMovieDataProcessor:
    """Enhanced movie data processor with multi-source support"""
    
    def __init__(self, data_source: str = None):
        """
        Initialize the enhanced movie processor
        
        Args:
            data_source: Data source type ("astra", "excel", "strapi", "sample")
        """
        self.data_source = data_source or config.DATA_SOURCE
        self.df = None
        self.vectorizer = None
        self.movie_vectors = None
        self.movie_chunks = []
        self.connector = None
        
        # Initialize based on data source
        self._initialize_data_source()
        self.load_and_process_data()
    
    def _initialize_data_source(self):
        """Initialize the appropriate data source connector"""
        try:
            if self.data_source.lower() == "astra":
                if AstraConnector is None:
                    logger.warning("Astra connector not available, falling back to sample data")
                    self.data_source = "sample"
                else:
                    astra_config = config.get_astra_config()
                    self.connector = AstraConnector(**astra_config)
                    logger.info("✅ Initialized Astra connector")
            
            elif self.data_source.lower() == "strapi":
                if StrapiConnector is None:
                    logger.warning("Strapi connector not available, falling back to sample data")
                    self.data_source = "sample"
                else:
                    # You would need to add Strapi config to config.py
                    self.connector = StrapiConnector(
                        base_url=os.getenv("STRAPI_URL", "http://localhost:1337"),
                        api_token=os.getenv("STRAPI_API_TOKEN"),
                        collection_name="movies"
                    )
                    logger.info("✅ Initialized Strapi connector")
            
            elif self.data_source.lower() == "excel":
                logger.info("✅ Using Excel data source")
            
            else:
                logger.info("✅ Using sample data source")
                
        except Exception as e:
            logger.error(f"❌ Error initializing data source: {e}")
            self.data_source = "sample"
    
    def load_and_process_data(self):
        """Load and process movie data from the configured source"""
        try:
            if self.data_source.lower() == "astra":
                self._load_from_astra()
            elif self.data_source.lower() == "strapi":
                self._load_from_strapi()
            elif self.data_source.lower() == "excel":
                self._load_from_excel()
            else:
                self._load_sample_data()
            
            # Process the data
            self.create_movie_chunks()
            self.create_vector_embeddings()
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            self._load_sample_data()
            self.create_movie_chunks()
            self.create_vector_embeddings()
    
    def _load_from_astra(self):
        """Load data from Astra database"""
        try:
            if not self.connector or not self.connector.test_connection():
                raise Exception("Astra connection failed")
            
            # Get all movies from Astra
            movies = self.connector.get_all_movies(limit=1000)
            
            if not movies:
                raise Exception("No movies found in Astra database")
            
            # Convert to DataFrame
            self.df = pd.DataFrame(movies)
            logger.info(f"✅ Loaded {len(self.df)} movies from Astra")
            
        except Exception as e:
            logger.error(f"❌ Error loading from Astra: {e}")
            raise
    
    def _load_from_strapi(self):
        """Load data from Strapi CMS"""
        try:
            if not self.connector or not self.connector.test_connection():
                raise Exception("Strapi connection failed")
            
            # Get all movies from Strapi
            movies = self.connector.get_all_movies(limit=1000)
            
            if not movies:
                raise Exception("No movies found in Strapi")
            
            # Transform Strapi data
            transformed_movies = self.connector.transform_strapi_data(movies)
            
            # Convert to DataFrame
            self.df = pd.DataFrame(transformed_movies)
            logger.info(f"✅ Loaded {len(self.df)} movies from Strapi")
            
        except Exception as e:
            logger.error(f"❌ Error loading from Strapi: {e}")
            raise
    
    def _load_from_excel(self):
        """Load data from Excel file"""
        try:
            if not os.path.exists(config.MOVIE_DATA_PATH):
                raise Exception(f"Excel file not found: {config.MOVIE_DATA_PATH}")
            
            self.df = pd.read_excel(config.MOVIE_DATA_PATH)
            self.df = self.df.fillna("Unknown")
            logger.info(f"✅ Loaded {len(self.df)} movies from Excel")
            
        except Exception as e:
            logger.error(f"❌ Error loading from Excel: {e}")
            raise
    
    def _load_sample_data(self):
        """Load sample data as fallback"""
        self.df = pd.DataFrame(config.SAMPLE_MOVIE_DATA)
        logger.info(f"✅ Loaded {len(self.df)} sample movies")
    
    def create_movie_chunks(self):
        """Create searchable chunks from movie data"""
        self.movie_chunks = []
        
        for _, row in self.df.iterrows():
            # Create multiple chunks per movie for better retrieval
            chunks = [
                f"Movie: {row['Movie Name']} ({row['Year']}) - {row['Description']}",
                f"Cast: {row['Movie Name']} features {row['Cast']}",
                f"Director: {row['Movie Name']} was directed by {row['Director']}",
                f"Details: {row['Movie Name']} is a {row['Genre']} film, {row['Runtime (min)']} minutes, rated {row['IMDb Rating']}/10",
                f"Business: {row['Movie Name']} grossed {row['Box Office']} at box office, available on {row['Streaming']}"
            ]
            
            for chunk in chunks:
                self.movie_chunks.append({
                    'text': chunk,
                    'movie_name': row['Movie Name'],
                    'year': row['Year'],
                    'genre': row['Genre'],
                    'rating': row['IMDb Rating']
                })
        
        logger.info(f"✅ Created {len(self.movie_chunks)} movie chunks")
    
    def create_vector_embeddings(self):
        """Create TF-IDF vectors for semantic search"""
        texts = [chunk['text'] for chunk in self.movie_chunks]
        
        search_config = config.get_search_config()
        self.vectorizer = TfidfVectorizer(
            stop_words=search_config['tfidf_stop_words'],
            max_features=search_config['tfidf_max_features']
        )
        self.movie_vectors = self.vectorizer.fit_transform(texts)
        
        logger.info(f"✅ Created vector embeddings for {len(texts)} chunks")
    
    def search_movies(self, query: str, top_k: int = None) -> List[Dict]:
        """Search movies using semantic similarity"""
        if top_k is None:
            top_k = config.DEFAULT_SEARCH_RESULTS
        
        # If using Astra, try direct database search first
        if self.data_source.lower() == "astra" and self.connector:
            try:
                db_results = self.connector.search_movies(query, limit=top_k)
                if db_results:
                    logger.info(f"✅ Found {len(db_results)} movies via Astra search")
                    return self._format_astra_results(db_results)
            except Exception as e:
                logger.warning(f"Database search failed, using vector search: {e}")
        
        # Fallback to vector search
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.movie_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        seen_movies = set()
        
        for idx in top_indices:
            chunk = self.movie_chunks[idx]
            if chunk['movie_name'] not in seen_movies:
                results.append({
                    'movie_name': chunk['movie_name'],
                    'year': chunk['year'],
                    'genre': chunk['genre'],
                    'rating': chunk['rating'],
                    'relevance_score': float(similarities[idx]),
                    'chunk_text': chunk['text']
                })
                seen_movies.add(chunk['movie_name'])
        
        logger.info(f"✅ Found {len(results)} movies via vector search")
        return results
    
    def _format_astra_results(self, db_results: List[Dict]) -> List[Dict]:
        """Format Astra database results to match expected format"""
        formatted_results = []
        
        for movie in db_results:
            formatted_results.append({
                'movie_name': movie.get('Movie Name', 'Unknown'),
                'year': movie.get('Year', 0),
                'genre': movie.get('Genre', 'Unknown'),
                'rating': movie.get('IMDb Rating', 0.0),
                'relevance_score': 1.0,  # Database results are considered highly relevant
                'chunk_text': f"Movie: {movie.get('Movie Name', 'Unknown')} ({movie.get('Year', 0)}) - {movie.get('Description', 'No description')}"
            })
        
        return formatted_results
    
    def get_movies_by_genre(self, genre: str, limit: int = 20) -> List[Dict]:
        """Get movies by genre"""
        if self.data_source.lower() == "astra" and self.connector:
            try:
                db_results = self.connector.get_movies_by_genre(genre, limit=limit)
                if db_results:
                    return self._format_astra_results(db_results)
            except Exception as e:
                logger.warning(f"Database genre search failed: {e}")
        
        # Fallback to DataFrame filtering
        genre_movies = self.df[self.df['Genre'].str.contains(genre, case=False, na=False)]
        genre_movies = genre_movies.nlargest(limit, 'IMDb Rating')
        
        results = []
        for _, movie in genre_movies.iterrows():
            results.append({
                'movie_name': movie['Movie Name'],
                'year': movie['Year'],
                'genre': movie['Genre'],
                'rating': movie['IMDb Rating'],
                'description': movie['Description']
            })
        
        return results
    
    def get_top_rated_movies(self, limit: int = 10) -> List[Dict]:
        """Get top-rated movies"""
        if self.data_source.lower() == "astra" and self.connector:
            try:
                db_results = self.connector.get_top_rated_movies(limit=limit)
                if db_results:
                    return self._format_astra_results(db_results)
            except Exception as e:
                logger.warning(f"Database top-rated search failed: {e}")
        
        # Fallback to DataFrame sorting
        top_movies = self.df.nlargest(limit, 'IMDb Rating')
        
        results = []
        for _, movie in top_movies.iterrows():
            results.append({
                'movie_name': movie['Movie Name'],
                'year': movie['Year'],
                'genre': movie['Genre'],
                'rating': movie['IMDb Rating'],
                'description': movie['Description']
            })
        
        return results
    
    def get_movie_statistics(self) -> Dict[str, Any]:
        """Get statistics about the movie database"""
        if self.data_source.lower() == "astra" and self.connector:
            try:
                return self.connector.get_movie_statistics()
            except Exception as e:
                logger.warning(f"Database statistics failed: {e}")
        
        # Fallback to DataFrame statistics
        stats = {
            'total_movies': len(self.df),
            'genres': self.df['Genre'].value_counts().to_dict(),
            'years': self.df['Year'].value_counts().to_dict(),
            'last_updated': datetime.now().isoformat(),
            'data_source': self.data_source
        }
        
        return stats
    
    def refresh_data(self):
        """Refresh data from the source"""
        logger.info("🔄 Refreshing movie data...")
        self.load_and_process_data()
        logger.info("✅ Data refresh completed")
    
    def close_connection(self):
        """Close database connection if applicable"""
        if self.connector and hasattr(self.connector, 'close'):
            try:
                self.connector.close()
                logger.info("✅ Database connection closed")
            except Exception as e:
                logger.error(f"❌ Error closing connection: {e}")

# Example usage
if __name__ == "__main__":
    # Test the enhanced processor
    processor = EnhancedMovieDataProcessor()
    
    # Test search
    results = processor.search_movies("action movies", top_k=5)
    print(f"Search results: {len(results)} movies")
    
    # Test statistics
    stats = processor.get_movie_statistics()
    print(f"Database statistics: {stats}")
    
    # Close connection
    processor.close_connection()
