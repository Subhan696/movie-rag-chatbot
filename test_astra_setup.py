"""
Test script to help configure Astra database connection
This will guide you through setting up your Astra credentials
"""

import os
import requests
from dotenv import load_dotenv

def test_astra_connection():
    """Test Astra connection with current configuration"""
    print("🔍 Testing Astra Database Connection...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    db_id = os.getenv("ASTRA_DB_ID")
    db_region = os.getenv("ASTRA_DB_REGION")
    client_id = os.getenv("ASTRA_CLIENT_ID")
    client_secret = os.getenv("ASTRA_CLIENT_SECRET")
    keyspace = os.getenv("ASTRA_KEYSPACE", "movies")
    table_name = os.getenv("ASTRA_TABLE_NAME", "movies")
    
    print(f"📋 Current Configuration:")
    print(f"   Database ID: {db_id or 'NOT SET'}")
    print(f"   Region: {db_region or 'NOT SET'}")
    print(f"   Client ID: {client_id or 'NOT SET'}")
    print(f"   Client Secret: {'SET' if client_secret else 'NOT SET'}")
    print(f"   Keyspace: {keyspace}")
    print(f"   Table: {table_name}")
    print()
    
    # Check if configuration is complete
    if not all([db_id, db_region, client_id, client_secret]):
        print("❌ Configuration incomplete!")
        print("\n📝 To set up your Astra database:")
        print("1. Go to https://astra.datastax.com/")
        print("2. Create a database and note the Database ID and Region")
        print("3. Create an Application Token and note the Client ID and Client Secret")
        print("4. Create a .env file with these values:")
        print()
        print("ASTRA_DB_ID=your_actual_database_id")
        print("ASTRA_DB_REGION=your_actual_region")
        print("ASTRA_CLIENT_ID=your_actual_client_id")
        print("ASTRA_CLIENT_SECRET=your_actual_client_secret")
        print("ASTRA_KEYSPACE=movies")
        print("ASTRA_TABLE_NAME=movies")
        print()
        return False
    
    # Test connection
    try:
        base_url = f"https://{db_id}-{db_region}.apps.astra.datastax.com/api/rest/v2/keyspaces/{keyspace}/{table_name}"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {client_secret}',
            'X-Cassandra-Token': client_secret
        }
        
        print(f"🌐 Testing connection to: {base_url}")
        
        # Try to get table schema first
        print("🔍 Checking table schema...")
        try:
            schema_response = requests.get(f"{base_url}/schema", headers=headers, timeout=10)
            print(f"   Schema response: {schema_response.status_code}")
            if schema_response.status_code == 200:
                schema_data = schema_response.json()
                print(f"   Table schema: {schema_data}")
            else:
                print(f"   Schema error: {schema_response.text}")
        except Exception as e:
            print(f"   Could not get schema: {e}")
        
        # Try a GET request with a simple where clause
        print("\n🔍 Testing data retrieval...")
        params = {
            'where': '{}',  # Empty where clause as JSON string
            'limit': 1
        }
        response = requests.get(base_url, headers=headers, params=params, timeout=10)
        print(f"   GET response: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Connection successful!")
            data = response.json()
            movie_count = len(data.get('data', []))
            print(f"📊 Found {movie_count} movies in database")
            return True
        elif response.status_code == 401:
            print("❌ Authentication failed!")
            print("   Check your Client ID and Client Secret")
            return False
        elif response.status_code == 404:
            print("❌ Table not found!")
            print("   Make sure the table 'ragcine' exists in your keyspace")
            return False
        else:
            print(f"❌ Connection failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
            
            # Try POST with where clause
            print("\n🔍 Trying POST method...")
            try:
                post_data = {
                    "where": {},
                    "limit": 1
                }
                post_response = requests.post(base_url, headers=headers, json=post_data, timeout=10)
                print(f"   POST response: {post_response.status_code}")
                if post_response.status_code == 200:
                    print("✅ POST method works!")
                    data = post_response.json()
                    movie_count = len(data.get('data', []))
                    print(f"📊 Found {movie_count} movies in database")
                    return True
                else:
                    print(f"   POST error: {post_response.text}")
            except Exception as e:
                print(f"   POST failed: {e}")
            
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error!")
        print("   Check your Database ID and Region")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection timeout!")
        print("   Check your internet connection")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def create_sample_env():
    """Create a sample .env file"""
    env_content = """# Astra Database Configuration
# Replace these with your actual values from Astra dashboard

# Your Database ID (found in Astra dashboard)
ASTRA_DB_ID=your_database_id_here

# Your Database Region (found in Astra dashboard)
ASTRA_DB_REGION=us-east1

# Your Application Token credentials (found in Settings > Application Tokens)
ASTRA_CLIENT_ID=your_client_id_here
ASTRA_CLIENT_SECRET=your_client_secret_here

# Database settings
ASTRA_KEYSPACE=movies
ASTRA_TABLE_NAME=movies

# Data source configuration
DATA_SOURCE=astra

# Gemini API (for the chatbot)
GEMINI_API_KEY=your_gemini_api_key_here
"""
    
    with open('.env.sample', 'w') as f:
        f.write(env_content)
    
    print("📄 Created .env.sample file")
    print("📝 Copy this to .env and fill in your actual values")

def main():
    """Main function"""
    print("🚀 Astra Database Setup Test")
    print("=" * 50)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("📝 Creating sample .env file...")
        create_sample_env()
        print("\n📋 Next steps:")
        print("1. Copy .env.sample to .env")
        print("2. Fill in your actual Astra credentials")
        print("3. Run this script again")
        return
    
    # Test connection
    success = test_astra_connection()
    
    if success:
        print("\n🎉 Astra connection is working!")
        print("✅ You can now run your movie chatbot")
        print("\n🚀 Next steps:")
        print("1. Load your movie data into Astra")
        print("2. Run: python movie_chatbot.py")
    else:
        print("\n🔧 Please fix the configuration issues above")
        print("📖 See ASTRA_SETUP.md for detailed instructions")

if __name__ == "__main__":
    main()
