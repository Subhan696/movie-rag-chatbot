# 🚀 Deployment Guide

This guide covers different ways to deploy your enhanced Movie RAG Chatbot.

## 📋 Prerequisites

- Python 3.8 or higher
- Gemini API key
- Movie data Excel file (optional - fallback data provided)

## 🏠 Local Development

### 1. Basic Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd movie-rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Run the chatbot
python movie_chatbot.py
```

### 2. Test the Installation
```bash
# Run tests to verify everything works
python test_chatbot.py
```

### 3. Customize Configuration
Edit `config.py` to customize:
- Search parameters
- UI settings
- File paths
- Validation rules

## 🌐 Web Deployment

### Option 1: Gradio Cloud (Recommended for Demo)

1. **Prepare for Gradio Cloud**
   ```bash
   # Ensure your code is in a GitHub repository
   git add .
   git commit -m "Enhanced RAG chatbot ready for deployment"
   git push origin main
   ```

2. **Deploy to Gradio Cloud**
   - Go to [Gradio Cloud](https://gradio.app/)
   - Connect your GitHub repository
   - Set environment variables:
     - `GEMINI_API_KEY`: Your Gemini API key
   - Deploy!

### Option 2: Heroku

1. **Create Heroku App**
   ```bash
   # Install Heroku CLI
   heroku create your-movie-chatbot
   ```

2. **Add Buildpacks**
   ```bash
   heroku buildpacks:add heroku/python
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set GEMINI_API_KEY=your_api_key_here
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

### Option 3: Railway

1. **Connect Repository**
   - Go to [Railway](https://railway.app/)
   - Connect your GitHub repository

2. **Configure Environment**
   - Add `GEMINI_API_KEY` environment variable
   - Set Python version to 3.8+

3. **Deploy**
   - Railway will automatically deploy your app

## 🐳 Docker Deployment

### 1. Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "movie_chatbot.py"]
```

### 2. Build and Run
```bash
# Build the Docker image
docker build -t movie-chatbot .

# Run the container
docker run -p 7860:7860 \
  -e GEMINI_API_KEY=your_api_key_here \
  movie-chatbot
```

### 3. Docker Compose (Recommended)
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  movie-chatbot:
    build: .
    ports:
      - "7860:7860"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## ☁️ Cloud Platform Deployment

### Google Cloud Run

1. **Create Dockerfile** (see above)

2. **Deploy to Cloud Run**
   ```bash
   # Build and push to Google Container Registry
   gcloud builds submit --tag gcr.io/PROJECT_ID/movie-chatbot
   
   # Deploy to Cloud Run
   gcloud run deploy movie-chatbot \
     --image gcr.io/PROJECT_ID/movie-chatbot \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars GEMINI_API_KEY=your_api_key_here
   ```

### AWS Lambda + API Gateway

1. **Create Lambda Function**
   ```python
   # lambda_function.py
   import json
   from movie_chatbot import demo
   
   def lambda_handler(event, context):
       # Handle API Gateway events
       return {
           'statusCode': 200,
           'body': json.dumps('Movie Chatbot API')
       }
   ```

2. **Deploy with Serverless Framework**
   ```yaml
   # serverless.yml
   service: movie-chatbot
   
   provider:
     name: aws
     runtime: python3.9
     region: us-east-1
     environment:
       GEMINI_API_KEY: ${env:GEMINI_API_KEY}
   
   functions:
     chatbot:
       handler: lambda_function.lambda_handler
       events:
         - http:
             path: /
             method: get
   ```

## 🔧 Production Considerations

### 1. Environment Variables
```bash
# Required
GEMINI_API_KEY=your_api_key_here

# Optional
MOVIE_DATA_PATH=/path/to/movies.xlsx
LOG_LEVEL=INFO
DEBUG=False
```

### 2. Data Management
- Store movie data in a database (PostgreSQL, MongoDB)
- Use cloud storage for large datasets
- Implement data caching for better performance

### 3. Security
- Use HTTPS in production
- Implement rate limiting
- Add authentication if needed
- Validate all user inputs

### 4. Monitoring
- Add logging to track usage
- Monitor API response times
- Set up error alerting
- Track user engagement metrics

### 5. Scaling
- Use load balancers for multiple instances
- Implement caching (Redis)
- Consider using a vector database (Pinecone, Weaviate)
- Use CDN for static assets

## 📊 Performance Optimization

### 1. Caching
```python
# Add Redis caching
import redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_response(query):
    return redis_client.get(f"response:{hash(query)}")
```

### 2. Database Integration
```python
# Use PostgreSQL for movie data
import psycopg2
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:pass@localhost/movies')
```

### 3. Vector Database
```python
# Use Pinecone for better vector search
import pinecone
pinecone.init(api_key='your-pinecone-key', environment='us-west1-gcp')
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```bash
   # Check environment variable
   echo $GEMINI_API_KEY
   
   # Or set it explicitly
   export GEMINI_API_KEY=your_key_here
   ```

2. **Port Already in Use**
   ```bash
   # Find process using port 7860
   lsof -i :7860
   
   # Kill the process
   kill -9 <PID>
   ```

3. **Memory Issues**
   ```bash
   # Increase memory for Docker
   docker run -m 2g movie-chatbot
   ```

4. **Dependencies Issues**
   ```bash
   # Clean install
   pip uninstall -r requirements.txt
   pip install -r requirements.txt
   ```

## 📈 Monitoring and Analytics

### 1. Add Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 2. Track Usage
```python
# Add analytics tracking
def track_usage(user_query, response_time):
    # Send to analytics service
    pass
```

### 3. Health Checks
```python
# Add health check endpoint
@app.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': datetime.now()}
```

## 🎯 Next Steps

1. **Add Authentication** - User login system
2. **Implement Caching** - Redis for performance
3. **Add Analytics** - Track usage patterns
4. **Scale Database** - Move to PostgreSQL/MongoDB
5. **Add CI/CD** - Automated testing and deployment
6. **Monitor Performance** - APM tools integration

---

**Need help?** Check the troubleshooting section or create an issue in the repository.
