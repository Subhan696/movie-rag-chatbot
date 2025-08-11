#!/usr/bin/env python3
"""
Test script for the enhanced Movie RAG Chatbot
Tests core functionality without requiring the full UI
"""

import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_movie_processor():
    """Test the MovieDataProcessor class"""
    print("🧪 Testing MovieDataProcessor...")
    
    try:
        from movie_chatbot import MovieDataProcessor
        
        # Test with sample data (since we don't have the actual Excel file)
        processor = MovieDataProcessor("nonexistent_file.xlsx")
        
        # Test data loading
        assert processor.df is not None, "DataFrame should be created"
        assert len(processor.df) > 0, "DataFrame should have data"
        print("✅ Data loading: PASSED")
        
        # Test chunking
        assert len(processor.movie_chunks) > 0, "Movie chunks should be created"
        print("✅ Movie chunking: PASSED")
        
        # Test vector embeddings
        assert processor.vectorizer is not None, "Vectorizer should be created"
        assert processor.movie_vectors is not None, "Movie vectors should be created"
        print("✅ Vector embeddings: PASSED")
        
        # Test search functionality
        results = processor.search_movies("action movies", top_k=3)
        assert len(results) > 0, "Search should return results"
        print("✅ Movie search: PASSED")
        
        print("🎉 MovieDataProcessor tests: ALL PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ MovieDataProcessor tests: FAILED - {e}\n")
        return False

def test_user_data_manager():
    """Test the UserDataManager class"""
    print("🧪 Testing UserDataManager...")
    
    try:
        from movie_chatbot import UserDataManager
        
        manager = UserDataManager("test_user_info.xlsx")
        
        # Test email validation
        assert manager.validate_email("test@example.com") == True, "Valid email should pass"
        assert manager.validate_email("invalid-email") == False, "Invalid email should fail"
        print("✅ Email validation: PASSED")
        
        # Test phone validation
        assert manager.validate_phone("1234567890") == True, "Valid phone should pass"
        assert manager.validate_phone("123") == False, "Invalid phone should fail"
        print("✅ Phone validation: PASSED")
        
        print("🎉 UserDataManager tests: ALL PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ UserDataManager tests: FAILED - {e}\n")
        return False

def test_session_manager():
    """Test the SessionManager class"""
    print("🧪 Testing SessionManager...")
    
    try:
        from movie_chatbot import SessionManager
        
        manager = SessionManager()
        
        # Test session initialization
        session = manager.init_session()
        assert session["name"] is None, "New session should have no name"
        assert session["collected"] == False, "New session should not be collected"
        assert "session_id" in session, "Session should have ID"
        print("✅ Session initialization: PASSED")
        
        # Test conversation log loading
        assert isinstance(manager.conversation_log, list), "Conversation log should be a list"
        print("✅ Conversation log loading: PASSED")
        
        print("🎉 SessionManager tests: ALL PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ SessionManager tests: FAILED - {e}\n")
        return False

def test_movie_recommender():
    """Test the MovieRecommender class"""
    print("🧪 Testing MovieRecommender...")
    
    try:
        from movie_chatbot import MovieDataProcessor, MovieRecommender
        
        processor = MovieDataProcessor("nonexistent_file.xlsx")
        recommender = MovieRecommender(processor)
        
        # Test recommendations
        recommendations = recommender.get_recommendations("drama movies", top_k=3)
        assert len(recommendations) > 0, "Should return recommendations"
        print("✅ Movie recommendations: PASSED")
        
        # Test genre recommendations
        genre_recs = recommender.get_genre_recommendations("Drama", top_k=3)
        assert len(genre_recs) > 0, "Should return genre recommendations"
        print("✅ Genre recommendations: PASSED")
        
        print("🎉 MovieRecommender tests: ALL PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ MovieRecommender tests: FAILED - {e}\n")
        return False

def test_utility_functions():
    """Test utility functions"""
    print("🧪 Testing utility functions...")
    
    try:
        from movie_chatbot import calculate_session_duration, export_conversation
        
        # Test session duration calculation
        session = {
            "start_time": datetime.now().isoformat(),
            "session_id": "test_session"
        }
        duration = calculate_session_duration(session)
        assert "minutes" in duration or duration == "Unknown", "Duration should be calculated"
        print("✅ Session duration calculation: PASSED")
        
        # Test conversation export
        session["chat_history"] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        result = export_conversation(session)
        assert "exported" in result.lower(), "Export should succeed"
        print("✅ Conversation export: PASSED")
        
        print("🎉 Utility function tests: ALL PASSED\n")
        return True
        
    except Exception as e:
        print(f"❌ Utility function tests: FAILED - {e}\n")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Enhanced Movie RAG Chatbot Tests\n")
    print("=" * 50)
    
    tests = [
        test_movie_processor,
        test_user_data_manager,
        test_session_manager,
        test_movie_recommender,
        test_utility_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The enhanced chatbot is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
