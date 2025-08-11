import os
import requests
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

print("🔍 Debugging Chatbot Data Loading...")
print("=" * 50)

# Check environment variables
db_id = os.getenv("ASTRA_DB_ID")
db_region = os.getenv("ASTRA_DB_REGION")
client_secret = os.getenv("ASTRA_CLIENT_SECRET")

print(f"DB ID: {db_id}")
print(f"Region: {db_region}")
print(f"Secret: {'✅ Set' if client_secret else '❌ Missing'}")

if not all([db_id, db_region, client_secret]):
    print("❌ Missing environment variables!")
    exit()

# Test connection
base_url = f"https://{db_id}-{db_region}.apps.astra.datastax.com/api/rest/v2/keyspaces/default_keyspace/ragcine"
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {client_secret}',
    'X-Cassandra-Token': client_secret
}

print(f"\n🔗 Testing connection to: {base_url}")

try:
    # Test connection
    params = {'where': '{}', 'limit': 1}
    response = requests.get(base_url, headers=headers, params=params, timeout=10)
    print(f"Connection status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Connection successful!")
        
        # Get sample data
        params = {'where': '{}', 'limit': 5}
        response = requests.get(base_url, headers=headers, params=params)
        data = response.json()
        records = data.get('data', [])
        
        print(f"📊 Found {len(records)} records")
        
        if records:
            print("\n📋 Sample record fields:")
            first_record = records[0]
            for key, value in first_record.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
            
            print("\n🎬 Sample movies:")
            for i, record in enumerate(records, 1):
                title = record.get('title', 'No title')
                rating = record.get('rating', 'N/A')
                print(f"  {i}. {title} (Rating: {rating})")
            
            # Test if we can create chunks
            print("\n🔍 Testing chunk creation...")
            df = pd.DataFrame(records)
            
            # Check for text fields
            text_fields = []
            text_field_names = ['text', 'title', 'rating', 'release_date', 'where_to_watch']
            
            for field in text_field_names:
                if field in df.columns:
                    non_null_count = df[field].notna().sum()
                    print(f"  {field}: {non_null_count}/{len(df)} records have data")
                    if non_null_count > 0:
                        text_fields.append(field)
            
            if text_fields:
                print(f"✅ Found {len(text_fields)} text fields: {text_fields}")
                print("🎯 Your chatbot should be able to load this data!")
            else:
                print("❌ No text fields found - this might be the problem")
                
        else:
            print("❌ No records found in database")
            
    else:
        print(f"❌ Connection failed: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n💡 If connection works but chatbot doesn't load data:")
print("   1. Check if the chatbot is using the right field names")
print("   2. Make sure the data has text content")
print("   3. Verify the chunk creation logic")
