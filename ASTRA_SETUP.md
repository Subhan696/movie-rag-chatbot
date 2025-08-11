# 🚀 DataStax Astra Database Integration Guide

This guide will help you set up and integrate your DataStax Astra database with the enhanced Movie RAG Chatbot.

## 📋 Prerequisites

- DataStax Astra account (free tier available)
- Python 3.8+
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
2. Download the **Secure Connect Bundle** (ZIP file)
3. Note your **Client ID** and **Client Secret**
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
# Install the required packages
pip install -r requirements.txt
```

## 🎯 Step 3: Configure Environment Variables

Create a `.env` file in your project root:

```bash
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Astra Database Configuration
ASTRA_SECURE_CONNECT_BUNDLE=path/to/secure-connect-database.zip
ASTRA_CLIENT_ID=your_astra_client_id
ASTRA_CLIENT_SECRET=your_astra_client_secret
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
        secure_connect_bundle_path=os.getenv("ASTRA_SECURE_CONNECT_BUNDLE"),
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

### Option B: Using CQL Commands

You can also insert data directly using CQL:

```sql
INSERT INTO movies.movies (
    id, title, year, description, director, cast, 
    genre, box_office, rating, streaming, runtime, 
    created_at, updated_at
) VALUES (
    uuid(), 'The Shawshank Redemption', 1994, 
    'Two imprisoned men bond over a number of years...',
    'Frank Darabont', 'Tim Robbins, Morgan Freeman',
    'Drama', '$58.3M', 9.3, 'Netflix', 142,
    toTimestamp(now()), toTimestamp(now())
);
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
   
   # Verify secure connect bundle exists
   ls -la secure-connect-database.zip
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
   - Check if your Astra user has the correct permissions
   - Ensure you're using the right keyspace

## 📊 Performance Optimization

### 1. Indexing for Better Search

Create secondary indexes for better performance:

```sql
-- Index for genre searches
CREATE INDEX ON movies.movies (genre);

-- Index for year searches
CREATE INDEX ON movies.movies (year);

-- Index for rating searches
CREATE INDEX ON movies.movies (rating);
```

### 2. Batch Operations

For large datasets, use batch operations:

```python
def batch_insert_movies(movies_data, batch_size=100):
    for i in range(0, len(movies_data), batch_size):
        batch = movies_data[i:i + batch_size]
        # Insert batch
        time.sleep(0.1)  # Rate limiting
```

### 3. Connection Pooling

The Cassandra driver automatically handles connection pooling, but you can configure it:

```python
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy

cluster = Cluster(
    cloud={'secure_connect_bundle': bundle_path},
    auth_provider=auth_provider,
    load_balancing_policy=DCAwareRoundRobinPolicy(local_dc='default'),
    protocol_version=4
)
```

## 🔒 Security Best Practices

1. **Environment Variables**: Never hardcode credentials
2. **Secure Connect Bundle**: Keep it secure and don't commit to version control
3. **Network Security**: Use Astra's built-in security features
4. **Access Control**: Use least-privilege access for your database user

## 📈 Monitoring and Analytics

### 1. Astra Metrics
- Monitor your database usage in the Astra dashboard
- Set up alerts for high usage
- Track query performance

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
export ASTRA_SECURE_CONNECT_BUNDLE=/app/secure-connect-database.zip
export ASTRA_CLIENT_ID=prod_client_id
export ASTRA_CLIENT_SECRET=prod_client_secret
export DATA_SOURCE=astra
```

### 2. Docker Configuration
```dockerfile
# Copy secure connect bundle
COPY secure-connect-database.zip /app/
ENV ASTRA_SECURE_CONNECT_BUNDLE=/app/secure-connect-database.zip
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
- **Astra Community**: [community.datastax.com](https://community.datastax.com/)
- **GitHub Issues**: Create an issue in this repository

---

**🎉 Congratulations!** Your Movie RAG Chatbot is now powered by DataStax Astra! 

The combination of Astra's scalable database with the enhanced RAG capabilities will provide:
- **Better Performance**: Fast, scalable database queries
- **Real-time Data**: Live updates to your movie database
- **Reliability**: Astra's cloud-native architecture
- **Scalability**: Handle millions of movies and users

Happy coding! 🎬✨
