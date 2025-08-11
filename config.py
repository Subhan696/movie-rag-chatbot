"""
Configuration file for the Movie RAG Chatbot
Easily customize settings without modifying the main code
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for the Movie RAG Chatbot"""
    
    # API Configuration
    GEMINI_MODEL = "gemini-2.0-flash-001"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # File Paths
    MOVIE_DATA_PATH = r"H:\\Subhan\\Hollywood_Top_Movies.xlsx"
    USER_INFO_FILE = "user_info.xlsx"
    CONVERSATION_LOG_FILE = "conversation_log.json"
    
    # Astra Database Configuration
    ASTRA_SECURE_CONNECT_BUNDLE = os.getenv("ASTRA_SECURE_CONNECT_BUNDLE", "secure-connect-database.zip")
    ASTRA_CLIENT_ID = os.getenv("ASTRA_CLIENT_ID")
    ASTRA_CLIENT_SECRET = os.getenv("ASTRA_CLIENT_SECRET")
    ASTRA_KEYSPACE = os.getenv("ASTRA_KEYSPACE", "default_keyspace")
    ASTRA_TABLE_NAME = os.getenv("ASTRA_TABLE_NAME", "ragcine")
    ASTRA_DB_ID = os.getenv("ASTRA_DB_ID", "your-db-id")
    ASTRA_DB_REGION = os.getenv("ASTRA_DB_REGION", "us-east1")
    
    # Data Source Configuration
    DATA_SOURCE = os.getenv("DATA_SOURCE", "astra")  # Options: "astra", "excel", "strapi"
    
    # Search Configuration
    DEFAULT_SEARCH_RESULTS = 3
    MAX_SEARCH_RESULTS = 10
    MIN_RECOMMENDATION_RATING = 7.0
    
    # UI Configuration
    CHATBOT_HEIGHT = 500
    THEME = "soft"
    TITLE = "Movie Expert Chatbot 🎬"
    
    # Session Configuration
    MAX_CONVERSATION_HISTORY = 100
    MAX_SESSION_DURATION_HOURS = 24
    
    # Validation Configuration
    MIN_PHONE_DIGITS = 10
    EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Vector Search Configuration
    TFIDF_MAX_FEATURES = 1000
    TFIDF_STOP_WORDS = 'english'
    
    # Chunking Configuration
    CHUNK_TYPES = [
        "movie_info",      # Basic movie information
        "cast_info",       # Cast and crew details
        "director_info",   # Director information
        "details_info",    # Technical details
        "business_info"    # Box office and streaming
    ]
    
    # Commands Configuration
    AVAILABLE_COMMANDS = {
        "help": "Show all available commands",
        "recommend": "Get movie recommendations",
        "top_rated": "Show highest-rated movies",
        "stats": "Show session statistics",
        "export": "Export conversation to file",
        "clear": "Clear current conversation"
    }
    
    # Error Messages
    ERROR_MESSAGES = {
        "api_error": "Sorry, I encountered an error while processing your request. Please try again later.",
        "data_error": "Sorry, I couldn't load the movie data. Please check the data source.",
        "validation_error": "Please provide valid information.",
        "unknown_command": "Unknown command. Type /help for available commands.",
        "database_error": "Sorry, I couldn't connect to the database. Please check your connection settings."
    }
    
    # Welcome Messages
    WELCOME_MESSAGE = """
    Hello! 👋
    Welcome to the Movie Expert Chatbot! 🎬
    May I have your name, please?
    """
    
    REGISTRATION_COMPLETE_MESSAGE = """
    Awesome, {name} from {location}! 🎉
    
    🎬 **You can now ask me about:**
    • Movie details, cast, directors
    • Box office performance and ratings
    • Streaming availability
    • Movie recommendations
    
    💡 **Try these commands:**
    • `/recommend action movies`
    • `/top rated`
    • `/export chat`
    
    Let's get started! 🍿
    """
    
    # Sample Data (fallback when database is not available)
    SAMPLE_MOVIE_DATA = {
        'Movie Name': [
            'The Shawshank Redemption',
            'The Godfather',
            'Pulp Fiction',
            'The Dark Knight',
            'Fight Club'
        ],
        'Year': [1994, 1972, 1994, 2008, 1999],
        'Description': [
            'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
            'The aging patriarch of an organized crime dynasty transfers control to his reluctant son.',
            'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine.',
            'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
            'An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.'
        ],
        'Director': [
            'Frank Darabont',
            'Francis Ford Coppola',
            'Quentin Tarantino',
            'Christopher Nolan',
            'David Fincher'
        ],
        'Cast': [
            'Tim Robbins, Morgan Freeman',
            'Marlon Brando, Al Pacino',
            'John Travolta, Samuel L. Jackson',
            'Christian Bale, Heath Ledger',
            'Brad Pitt, Edward Norton'
        ],
        'Genre': [
            'Drama',
            'Crime, Drama',
            'Crime, Drama',
            'Action, Crime, Drama',
            'Drama'
        ],
        'Box Office': [
            '$58.3M',
            '$245M',
            '$213.9M',
            '$1.005B',
            '$100.9M'
        ],
        'IMDb Rating': [9.3, 9.2, 8.9, 9.0, 8.8],
        'Streaming': [
            'Netflix',
            'Amazon Prime',
            'Hulu',
            'HBO Max',
            'Netflix'
        ],
        'Runtime (min)': [142, 175, 154, 152, 139]
    }
    
    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """Get search configuration"""
        return {
            "default_results": cls.DEFAULT_SEARCH_RESULTS,
            "max_results": cls.MAX_SEARCH_RESULTS,
            "min_rating": cls.MIN_RECOMMENDATION_RATING,
            "tfidf_max_features": cls.TFIDF_MAX_FEATURES,
            "tfidf_stop_words": cls.TFIDF_STOP_WORDS
        }
    
    @classmethod
    def get_ui_config(cls) -> Dict[str, Any]:
        """Get UI configuration"""
        return {
            "height": cls.CHATBOT_HEIGHT,
            "theme": cls.THEME,
            "title": cls.TITLE
        }
    
    @classmethod
    def get_validation_config(cls) -> Dict[str, Any]:
        """Get validation configuration"""
        return {
            "min_phone_digits": cls.MIN_PHONE_DIGITS,
            "email_regex": cls.EMAIL_REGEX
        }
    
    @classmethod
    def get_astra_config(cls) -> Dict[str, Any]:
        """Get Astra database configuration"""
        return {
            "secure_connect_bundle_path": cls.ASTRA_SECURE_CONNECT_BUNDLE,
            "client_id": cls.ASTRA_CLIENT_ID,
            "client_secret": cls.ASTRA_CLIENT_SECRET,
            "keyspace": cls.ASTRA_KEYSPACE,
            "table_name": cls.ASTRA_TABLE_NAME,
            "db_id": cls.ASTRA_DB_ID,
            "db_region": cls.ASTRA_DB_REGION
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        errors = []
        
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is not set")
        
        if cls.DEFAULT_SEARCH_RESULTS > cls.MAX_SEARCH_RESULTS:
            errors.append("DEFAULT_SEARCH_RESULTS cannot be greater than MAX_SEARCH_RESULTS")
        
        if cls.MIN_RECOMMENDATION_RATING < 0 or cls.MIN_RECOMMENDATION_RATING > 10:
            errors.append("MIN_RECOMMENDATION_RATING must be between 0 and 10")
        
        # Validate Astra configuration if using Astra
        if cls.DATA_SOURCE.lower() == "astra":
            if not cls.ASTRA_CLIENT_ID:
                errors.append("ASTRA_CLIENT_ID is not set")
            if not cls.ASTRA_CLIENT_SECRET:
                errors.append("ASTRA_CLIENT_SECRET is not set")
            if not os.path.exists(cls.ASTRA_SECURE_CONNECT_BUNDLE):
                errors.append(f"ASTRA_SECURE_CONNECT_BUNDLE not found: {cls.ASTRA_SECURE_CONNECT_BUNDLE}")
        
        if errors:
            print("Configuration errors found:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

# Global configuration instance
config = Config()
