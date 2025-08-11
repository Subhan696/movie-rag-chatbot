# 🚀 DataStax Astra Database Integration Guide (REST API)

This guide will help you set up and integrate your DataStax Astra database with the enhanced Movie RAG Chatbot using the REST API approach.

## 📋 Prerequisites

- DataStax Astra account (free tier available)
- Python 3.8+ (including Python 3.13)
- Movie data in your Astra database

## 🎯 Step 1: Set Up DataStax Astra

### 1.1 Create Astra Database
1. Go to [DataStax Astra](https://astra.datastax.com/)
2. Sign up for a free account
3. Create a new database:
   - Choose a database name (e.g., "movie-chatbot")
   - Select a keyspace name (e.g., "movies")
   - Choose a cloud provider and region
   - Click "Create Database"

### 1.2 Get Connection Details
1. In your Astra dashboard, go to "Connect"
2. Note your **Database ID** and **Region** from the connection details
3. Create an **Application Token**:
   - Go to "Settings" → "Application Tokens"
   - Click "Generate Token"
   - Save the **Client ID** and **Client Secret**
4. Save these credentials securely

### 1.3 Create Movie Table
Run this CQL command in Astra's CQL Console:

```sql
CREATE TABLE IF NOT EXISTS movies.movies (
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
);
```

## 🎯 Step 2: Install Dependencies

```bash
# Install the required packages (no Cassandra driver needed!)
pip install -r requirements.txt
```

## 🎯 Step 3: Configure Environment Variables

Create a `.env` file in your project root:

```bash
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Astra Database Configuration (REST API)
ASTRA_DB_ID=your_database_id_here
ASTRA_DB_REGION=your_region_here
ASTRA_CLIENT_ID=your_client_id_here
ASTRA_CLIENT_SECRET=your_client_secret_here
ASTRA_KEYSPACE=movies
ASTRA_TABLE_NAME=movies

# Data Source Configuration
DATA_SOURCE=astra
```

## 🎯 Step 4: Load Movie Data into Astra

### Option A: Using the Astra Connector Script

Create a data loader script:

```python
# load_movies_to_astra.py
import os
import pandas as pd
from astra_connector import AstraConnector
from dotenv import load_dotenv

load_dotenv()

def load_movies_to_astra():
    # Initialize Astra connector
    connector = AstraConnector(
        secure_connect_bundle_path="dummy",  # Not used in REST API
        client_id=os.getenv("ASTRA_CLIENT_ID"),
        client_secret=os.getenv("ASTRA_CLIENT_SECRET"),
        keyspace=os.getenv("ASTRA_KEYSPACE", "movies"),
        table_name=os.getenv("ASTRA_TABLE_NAME", "movies")
    )
    
    # Test connection
    if not connector.test_connection():
        print("❌ Failed to connect to Astra")
        return
    
    # Load your movie data (from Excel, CSV, or API)
    # Example: Load from Excel file
    df = pd.read_excel("your_movies.xlsx")
    
    # Insert movies into Astra
    success_count = 0
    for _, row in df.iterrows():
        movie_data = {
            'title': row['Movie Name'],
            'year': row['Year'],
            'description': row['Description'],
            'director': row['Director'],
            'cast': row['Cast'],
            'genre': row['Genre'],
            'box_office': row['Box Office'],
            'rating': row['IMDb Rating'],
            'streaming': row['Streaming'],
            'runtime': row['Runtime (min)']
        }
        
        if connector.insert_movie(movie_data):
            success_count += 1
    
    print(f"✅ Successfully loaded {success_count} movies to Astra")
    connector.close()

if __name__ == "__main__":
    load_movies_to_astra()
```

### Option B: Using REST API Directly

You can also insert data using the REST API:

```bash
# Example using curl
curl -X POST "https://your-db-id-your-region.apps.astra.datastax.com/api/rest/v2/keyspaces/movies/movies" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_client_secret" \
  -d '{
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "The Shawshank Redemption",
    "year": 1994,
    "description": "Two imprisoned men bond over a number of years...",
    "director": "Frank Darabont",
    "cast": "Tim Robbins, Morgan Freeman",
    "genre": "Drama",
    "box_office": "$58.3M",
    "rating": 9.3,
    "streaming": "Netflix",
    "runtime": 142,
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  }'
```

## 🎯 Step 5: Test the Integration

### 5.1 Test Astra Connection

```bash
python astra_connector.py
```

### 5.2 Test Enhanced Processor

```bash
python enhanced_movie_processor.py
```

### 5.3 Test Full Chatbot

```bash
python movie_chatbot.py
```

## 🎯 Step 6: Verify Everything Works

1. **Start the chatbot** and complete user registration
2. **Test basic queries**:
   - "Who directed The Shawshank Redemption?"
   - "What movies feature Morgan Freeman?"
   - "Show me action movies from 1994"

3. **Test advanced commands**:
   - `/top rated`
   - `/recommend drama movies`
   - `/stats`

## 🔧 Troubleshooting

### Common Issues

1. **Connection Failed**
   ```bash
   # Check your credentials
   echo $ASTRA_CLIENT_ID
   echo $ASTRA_CLIENT_SECRET
   echo $ASTRA_DB_ID
   echo $ASTRA_DB_REGION
   ```

2. **Table Not Found**
   ```sql
   -- Check if table exists
   DESCRIBE movies.movies;
   
   -- Create table if missing
   CREATE TABLE IF NOT EXISTS movies.movies (...);
   ```

3. **No Data Found**
   ```python
   # Test data retrieval
   from astra_connector import AstraConnector
   
   connector = AstraConnector(...)
   movies = connector.get_all_movies(limit=5)
   print(f"Found {len(movies)} movies")
   ```

4. **Permission Denied**
   - Check if your Application Token has the correct permissions
   - Ensure you're using the right keyspace

## 📊 Performance Optimization

### 1. REST API Best Practices

- Use appropriate HTTP methods (GET, POST, PUT, DELETE)
- Implement proper error handling
- Use pagination for large datasets
- Cache frequently accessed data

### 2. Batch Operations

For large datasets, use batch operations:

```python
def batch_insert_movies(movies_data, batch_size=100):
    for i in range(0, len(movies_data), batch_size):
        batch = movies_data[i:i + batch_size]
        # Insert batch using REST API
        time.sleep(0.1)  # Rate limiting
```

### 3. Connection Management

The REST API approach doesn't require connection pooling, but you can optimize:

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)
```

## 🔒 Security Best Practices

1. **Environment Variables**: Never hardcode credentials
2. **Application Tokens**: Use least-privilege access
3. **Network Security**: Use HTTPS for all API calls
4. **Token Rotation**: Regularly rotate your application tokens

## 📈 Monitoring and Analytics

### 1. Astra Metrics
- Monitor your database usage in the Astra dashboard
- Set up alerts for high usage
- Track API request performance

### 2. Application Metrics
```python
# Add timing to your queries
import time

start_time = time.time()
results = connector.search_movies("action")
query_time = time.time() - start_time
print(f"Query took {query_time:.2f} seconds")
```

## 🚀 Production Deployment

### 1. Environment Setup
```bash
# Production environment variables
export ASTRA_DB_ID=prod_db_id
export ASTRA_DB_REGION=prod_region
export ASTRA_CLIENT_ID=prod_client_id
export ASTRA_CLIENT_SECRET=prod_client_secret
export DATA_SOURCE=astra
```

### 2. Docker Configuration
```dockerfile
# No need for secure connect bundle
ENV ASTRA_DB_ID=your_db_id
ENV ASTRA_DB_REGION=your_region
ENV ASTRA_CLIENT_ID=your_client_id
ENV ASTRA_CLIENT_SECRET=your_client_secret
```

### 3. Health Checks
```python
def health_check():
    try:
        connector = AstraConnector(...)
        return connector.test_connection()
    except:
        return False
```

## 🎯 Next Steps

1. **Scale Your Database**: Upgrade to paid tier for more resources
2. **Add Real-time Updates**: Set up webhooks for data changes
3. **Implement Caching**: Add Redis for frequently accessed data
4. **Add Analytics**: Track user queries and popular movies
5. **Multi-region**: Deploy across multiple regions for better performance

## 📞 Support

- **Astra Documentation**: [docs.datastax.com](https://docs.datastax.com/)
- **Astra REST API Docs**: [docs.datastax.com/en/astra/docs/rest-api.html](https://docs.datastax.com/en/astra/docs/rest-api.html)
- **Astra Community**: [community.datastax.com](https://community.datastax.com/)
- **GitHub Issues**: Create an issue in this repository

---

**🎉 Congratulations!** Your Movie RAG Chatbot is now powered by DataStax Astra using REST API! 

The combination of Astra's scalable database with the enhanced RAG capabilities will provide:
- **Better Performance**: Fast, scalable REST API queries
- **Real-time Data**: Live updates to your movie database
- **Reliability**: Astra's cloud-native architecture
- **Scalability**: Handle millions of movies and users
- **Python 3.13 Compatibility**: No more Cassandra driver issues!

Happy coding! 🎬✨
