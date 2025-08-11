# 🎬 Movie Expert RAG Chatbot

An intelligent, interactive chatbot that provides accurate and conversational insights about Hollywood movies using advanced RAG (Retrieval-Augmented Generation) technology.

## 🚀 Enhanced Features

### Core RAG Capabilities
* ✅ **Advanced Vector Search** - TF-IDF embeddings for semantic movie search
* 🤖 **Gemini 2.0 Flash Integration** - Latest AI model for intelligent responses
* 🔍 **Smart Chunking** - Multiple searchable chunks per movie for better retrieval
* 📊 **Dynamic Data Processing** - Handles missing data gracefully with sample fallback
* 🧾 **Enhanced User Management** - Email/phone validation and session tracking

### User Experience
* 💬 **Conversational Interface** - Natural, engaging conversations with personalization
* 🎯 **Smart Recommendations** - AI-powered movie suggestions based on preferences
* 📈 **Session Analytics** - Track conversation statistics and duration
* 💾 **Conversation Export** - Save chat history to text files
* 🔄 **Command System** - Special commands for quick access to features

### Advanced Features
* 🎨 **Modern UI** - Beautiful Gradio interface with Soft theme
* 📱 **Responsive Design** - Works on desktop and mobile
* 🛡️ **Error Handling** - Robust error management and graceful degradation
* 📝 **Conversation Logging** - Persistent conversation history
* 🎭 **Multi-modal Support** - Ready for future image/movie poster integration

## 🧠 How It Works (Enhanced RAG Flow)

1. **Data Processing**: Loads movie data and creates semantic chunks
2. **Vector Embeddings**: Generates TF-IDF vectors for semantic search
3. **Query Processing**: Analyzes user input and extracts intent
4. **Smart Retrieval**: Finds most relevant movie information using similarity search
5. **Context Building**: Combines relevant data with conversation history
6. **AI Generation**: Uses Gemini 2.0 Flash for intelligent, contextual responses
7. **Response Enhancement**: Adds personalization and formatting

## 🧰 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **AI Model** | Gemini 2.0 Flash | Intelligent response generation |
| **Vector Search** | TF-IDF + Cosine Similarity | Semantic movie retrieval |
| **UI Framework** | Gradio 4.0+ | Modern web interface |
| **Data Processing** | Pandas + Scikit-learn | Data handling and ML |
| **Session Management** | Custom Classes | User state and conversation tracking |
| **Validation** | Regex + Custom Logic | Input validation and sanitization |

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- Gemini API key

### Installation
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd movie-rag-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

4. **Run the chatbot**
   ```bash
   python movie_chatbot.py
   ```

## 🎯 Usage Examples

### Basic Queries
```
User: "Who directed Titanic?"
Bot: "James Cameron directed Titanic (1997). It's a romantic drama about..."

User: "What movies feature Leonardo DiCaprio?"
Bot: "Leonardo DiCaprio has starred in several notable films including..."

User: "Show me action movies from 1994"
Bot: "Here are some action movies from 1994: Pulp Fiction, Speed..."
```

### Advanced Commands
```
/recommend sci-fi movies     # Get personalized recommendations
/top rated                  # Show highest-rated movies
/stats                      # View session statistics
/export chat               # Export conversation
/help                      # Show all commands
```

### Recommendation System
```
User: "I like movies with plot twists"
Bot: "🎬 Movie Recommendations for 'movies with plot twists':
1. The Shawshank Redemption (1994) - ⭐ 9.3/10
   Genre: Drama
2. Pulp Fiction (1994) - ⭐ 8.9/10
   Genre: Crime, Drama
..."
```

## 📊 Features in Detail

### Smart Data Processing
- **Automatic Chunking**: Creates multiple searchable chunks per movie
- **Semantic Search**: Uses TF-IDF vectors for intelligent retrieval
- **Data Validation**: Handles missing or corrupted data gracefully
- **Sample Data**: Provides fallback data for testing

### Enhanced User Experience
- **Personalization**: Remembers user preferences and location
- **Conversation Memory**: Maintains context across interactions
- **Input Validation**: Validates email and phone formats
- **Error Recovery**: Graceful handling of API failures

### Advanced Analytics
- **Session Tracking**: Monitor conversation duration and interaction count
- **Usage Statistics**: Track popular queries and user engagement
- **Export Capabilities**: Save conversations for analysis
- **Performance Metrics**: Monitor response times and accuracy

## 🔧 Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### Data Sources
- **Primary**: Excel file with movie data
- **Fallback**: Built-in sample data for testing
- **User Data**: Excel file for user registration

### Customization Options
- Adjust `top_k` parameter for search results
- Modify chunking strategy in `create_movie_chunks()`
- Customize UI theme and layout
- Add new command handlers

## 🚀 Future Enhancements

### Planned Features
- [ ] **Movie Poster Integration** - Display movie images
- [ ] **Voice Input/Output** - Speech-to-text and text-to-speech
- [ ] **Multi-language Support** - Internationalization
- [ ] **Advanced Analytics Dashboard** - Usage insights
- [ ] **Real-time Movie Data** - API integration for latest info
- [ ] **Social Features** - Share recommendations and reviews

### Technical Improvements
- [ ] **Vector Database** - Pinecone/Weaviate integration
- [ ] **Caching Layer** - Redis for performance
- [ ] **API Endpoints** - RESTful API for external access
- [ ] **Docker Support** - Containerized deployment
- [ ] **CI/CD Pipeline** - Automated testing and deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini for AI capabilities
- Gradio for the beautiful UI framework
- The movie data community for datasets
- All contributors and users

---

**Ready to explore the world of movies with AI? Start chatting now! 🎬✨**



