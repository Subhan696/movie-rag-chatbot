# 📦 Import necessary libraries
import os
import pandas as pd
import google.generativeai as genai  # ✅ Gemini SDK
from dotenv import load_dotenv
import gradio as gr
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import time
from datetime import datetime
import re
from typing import List, Dict, Any, Optional

# 🔷 Load Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 📁 Paths
MOVIE_PATH = r"H:\\Subhan\\Hollywood_Top_Movies.xlsx"
EXCEL_FILE = "user_info.xlsx"
CONVERSATION_LOG = "conversation_log.json"

# 🔷 Enhanced Movie Data Processing
class MovieDataProcessor:
    def __init__(self, movie_path: str):
        self.movie_path = movie_path
        self.df = None
        self.vectorizer = None
        self.movie_vectors = None
        self.movie_chunks = []
        self.load_and_process_data()
    
    def load_and_process_data(self):
        """Load and process movie data with enhanced features"""
        try:
            self.df = pd.read_excel(self.movie_path)
            self.df = self.df.fillna("Unknown")
            self.create_movie_chunks()
            self.create_vector_embeddings()
        except Exception as e:
            print(f"Error loading movie data: {e}")
            # Create sample data if file not found
            self.df = self.create_sample_data()
            self.create_movie_chunks()
            self.create_vector_embeddings()
    
    def create_sample_data(self):
        """Create sample movie data for testing"""
        sample_data = {
            'Movie Name': ['The Shawshank Redemption', 'The Godfather', 'Pulp Fiction'],
            'Year': [1994, 1972, 1994],
            'Description': ['Two imprisoned men bond over a number of years...', 'The aging patriarch of an organized crime dynasty...', 'The lives of two mob hitmen, a boxer...'],
            'Director': ['Frank Darabont', 'Francis Ford Coppola', 'Quentin Tarantino'],
            'Cast': ['Tim Robbins, Morgan Freeman', 'Marlon Brando, Al Pacino', 'John Travolta, Samuel L. Jackson'],
            'Genre': ['Drama', 'Crime, Drama', 'Crime, Drama'],
            'Box Office': ['$58.3M', '$245M', '$213.9M'],
            'IMDb Rating': [9.3, 9.2, 8.9],
            'Streaming': ['Netflix', 'Amazon Prime', 'Hulu'],
            'Runtime (min)': [142, 175, 154]
        }
        return pd.DataFrame(sample_data)
    
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
    
    def create_vector_embeddings(self):
        """Create TF-IDF vectors for semantic search"""
        texts = [chunk['text'] for chunk in self.movie_chunks]
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.movie_vectors = self.vectorizer.fit_transform(texts)
    
    def search_movies(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search movies using semantic similarity"""
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
        
        return results

# 🔷 Enhanced Session Management
class SessionManager:
    def __init__(self):
        self.conversation_log = self.load_conversation_log()
    
    def init_session(self):
        return {
            "name": None, "phone": None, "email": None, "location": None,
            "collected": False,
            "chat_history": [],
            "first_prompt": True,
            "session_id": f"session_{int(time.time())}",
            "start_time": datetime.now().isoformat(),
            "interaction_count": 0
        }
    
    def load_conversation_log(self):
        """Load conversation history from file"""
        try:
            if os.path.exists(CONVERSATION_LOG):
                with open(CONVERSATION_LOG, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading conversation log: {e}")
        return []
    
    def save_conversation_log(self, session):
        """Save conversation to log file"""
        try:
            log_entry = {
                "session_id": session["session_id"],
                "user_name": session.get("name", "Anonymous"),
                "start_time": session["start_time"],
                "end_time": datetime.now().isoformat(),
                "interaction_count": session["interaction_count"],
                "chat_history": session["chat_history"]
            }
            
            self.conversation_log.append(log_entry)
            
            # Keep only last 100 conversations
            if len(self.conversation_log) > 100:
                self.conversation_log = self.conversation_log[-100:]
            
            with open(CONVERSATION_LOG, 'w') as f:
                json.dump(self.conversation_log, f, indent=2)
        except Exception as e:
            print(f"Error saving conversation log: {e}")

# 🔷 Enhanced User Data Management
class UserDataManager:
    def __init__(self, excel_file: str):
        self.excel_file = excel_file
    
    def save_user_info(self, session):
        """Save user information with enhanced validation"""
        try:
            user_info = {
                "Name": session["name"],
                "Phone": session["phone"],
                "Email": session["email"],
                "Location": session["location"],
                "Registration_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Session_ID": session["session_id"]
            }
            
            if os.path.exists(self.excel_file):
                df = pd.read_excel(self.excel_file)
                df = pd.concat([df, pd.DataFrame([user_info])], ignore_index=True)
            else:
                df = pd.DataFrame([user_info])
            
            df.to_excel(self.excel_file, index=False)
            return True
        except Exception as e:
            print(f"Error saving user info: {e}")
            return False
    
    def validate_email(self, email: str) -> bool:
        """Basic email validation"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_phone(self, phone: str) -> bool:
        """Basic phone validation"""
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', phone)
        return len(digits_only) >= 10

# 🔷 Enhanced Movie Recommendations
class MovieRecommender:
    def __init__(self, movie_processor: MovieDataProcessor):
        self.movie_processor = movie_processor
    
    def get_recommendations(self, user_preferences: str, top_k: int = 5) -> List[Dict]:
        """Get movie recommendations based on user preferences"""
        # Search for movies similar to user preferences
        similar_movies = self.movie_processor.search_movies(user_preferences, top_k * 2)
        
        # Filter and rank by rating
        recommendations = []
        for movie in similar_movies:
            if movie['rating'] >= 7.0:  # Only recommend well-rated movies
                recommendations.append(movie)
        
        return recommendations[:top_k]
    
    def get_genre_recommendations(self, genre: str, top_k: int = 5) -> List[Dict]:
        """Get top movies by genre"""
        genre_movies = []
        for _, row in self.movie_processor.df.iterrows():
            if genre.lower() in row['Genre'].lower():
                genre_movies.append({
                    'movie_name': row['Movie Name'],
                    'year': row['Year'],
                    'genre': row['Genre'],
                    'rating': row['IMDb Rating'],
                    'description': row['Description']
                })
        
        # Sort by rating and return top k
        genre_movies.sort(key=lambda x: x['rating'], reverse=True)
        return genre_movies[:top_k]

# 🔷 Initialize components
movie_processor = MovieDataProcessor(MOVIE_PATH)
session_manager = SessionManager()
user_data_manager = UserDataManager(EXCEL_FILE)
movie_recommender = MovieRecommender(movie_processor)

# 🔷 Enhanced Gemini query with better RAG
def query_gemini(user_query: str, session: Dict, relevant_movies: List[Dict] = None) -> str:
    """Enhanced query function with better RAG and error handling"""
    user_name = session.get("name", "User")
    user_location = session.get("location", "their location")
    
    # Get relevant movies if not provided
    if relevant_movies is None:
        relevant_movies = movie_processor.search_movies(user_query, top_k=3)
    
    # Create enhanced context
    movie_context = ""
    for movie in relevant_movies:
        movie_context += f"• {movie['chunk_text']}\n"
    
    system_message = (
        f"You are a knowledgeable and friendly movie expert assistant. "
        f"The user you're helping is named {user_name} from {user_location}. "
        f"Make your responses conversational, informative, and engaging. "
        f"Use the provided movie data to answer questions accurately. "
        f"If you don't have information about something, say so clearly. "
        f"Keep responses concise but informative."
    )
    
    # Get recent chat history for context
    recent_history = session["chat_history"][-3:]  # Last 3 exchanges
    history_text = ""
    for turn in recent_history:
        role = turn["role"].capitalize()
        history_text += f"{role}: {turn['content']}\n"
    
    prompt = (
        f"{system_message}\n\n"
        f"### RECENT CONVERSATION:\n{history_text}\n"
        f"### RELEVANT MOVIE INFORMATION:\n{movie_context}\n"
        f"### USER QUESTION:\n{user_query}\n\n"
        f"Please provide a helpful and accurate response based on the movie information provided."
    )
    
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-001")
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Sorry {user_name}, I encountered an error while processing your request. Please try again later. (Error: {str(e)})"

# 🔷 Enhanced input handler
def handle_input(user_input: str, session: Dict) -> tuple[str, Dict]:
    """Enhanced input handler with better flow control and features"""
    session["interaction_count"] += 1
    
    if session["first_prompt"]:
        session["first_prompt"] = False
        return "Hello! 👋\nWelcome to the Movie Expert Chatbot! 🎬\nMay I have your name, please?", session
    
    if not session["collected"]:
        return handle_user_registration(user_input, session)
    
    # Handle special commands
    if user_input.lower().startswith('/'):
        return handle_special_commands(user_input, session)
    
    # Regular movie query
    session["chat_history"].append({"role": "user", "content": user_input})
    
    # Check if it's a recommendation request
    if any(word in user_input.lower() for word in ['recommend', 'suggestion', 'similar', 'like']):
        response = handle_recommendation_request(user_input, session)
    else:
        # Regular movie query
        relevant_movies = movie_processor.search_movies(user_input, top_k=3)
        response = query_gemini(user_input, session, relevant_movies)
    
    session["chat_history"].append({"role": "assistant", "content": response})
    return response, session

def handle_user_registration(user_input: str, session: Dict) -> tuple[str, Dict]:
    """Handle user registration flow with validation"""
    if not session["name"]:
        session["name"] = user_input.strip()
        return "Thank you! 📱 May I have your phone number?", session
    elif not session["phone"]:
        if not user_data_manager.validate_phone(user_input):
            return "Please enter a valid phone number (at least 10 digits).", session
        session["phone"] = user_input.strip()
        return "Great! 📧 May I have your email address?", session
    elif not session["email"]:
        if not user_data_manager.validate_email(user_input):
            return "Please enter a valid email address.", session
        session["email"] = user_input.strip()
        return "Thank you! 🌍 May I know your location?", session
    elif not session["location"]:
        session["location"] = user_input.strip()
        session["collected"] = True
        if user_data_manager.save_user_info(session):
            return (
                f"Awesome, {session['name']} from {session['location']}! 🎉\n\n"
                f"🎬 **You can now ask me about:**\n"
                f"• Movie details, cast, directors\n"
                f"• Box office performance and ratings\n"
                f"• Streaming availability\n"
                f"• Movie recommendations\n\n"
                f"💡 **Try these commands:**\n"
                f"• `/recommend action movies`\n"
                f"• `/top rated`\n"
                f"• `/export chat`\n\n"
                f"Let's get started! 🍿"
            ), session
        else:
            return "Registration completed, but there was an issue saving your info. You can still use the chatbot!", session

def handle_special_commands(user_input: str, session: Dict) -> tuple[str, Dict]:
    """Handle special commands"""
    command = user_input.lower().strip()
    
    if command == '/help':
        return (
            "🎬 **Available Commands:**\n\n"
            "• `/recommend [genre/preference]` - Get movie recommendations\n"
            "• `/top rated` - Show top-rated movies\n"
            "• `/export chat` - Export conversation\n"
            "• `/stats` - Show conversation statistics\n"
            "• `/help` - Show this help message\n\n"
            "Just ask normal questions about movies too!"
        ), session
    
    elif command.startswith('/recommend'):
        return handle_recommendation_request(user_input, session)
    
    elif command == '/top rated':
        top_movies = movie_processor.df.nlargest(5, 'IMDb Rating')[['Movie Name', 'Year', 'IMDb Rating', 'Genre']]
        response = "🏆 **Top 5 Highest Rated Movies:**\n\n"
        for _, movie in top_movies.iterrows():
            response += f"• **{movie['Movie Name']}** ({movie['Year']}) - ⭐ {movie['IMDb Rating']}/10 - {movie['Genre']}\n"
        return response, session
    
    elif command == '/stats':
        return (
            f"📊 **Conversation Statistics:**\n\n"
            f"• Session ID: {session['session_id']}\n"
            f"• Interactions: {session['interaction_count']}\n"
            f"• Session Duration: {calculate_session_duration(session)}\n"
            f"• User: {session.get('name', 'Anonymous')}\n"
            f"• Location: {session.get('location', 'Unknown')}"
        ), session
    
    elif command == '/export chat':
        return export_conversation(session), session
    
    else:
        return "Unknown command. Type `/help` for available commands.", session

def handle_recommendation_request(user_input: str, session: Dict) -> str:
    """Handle movie recommendation requests"""
    # Extract preference from command
    if user_input.lower().startswith('/recommend'):
        preference = user_input[11:].strip()  # Remove '/recommend '
    else:
        preference = user_input
    
    if not preference:
        preference = "popular movies"
    
    # Get recommendations
    recommendations = movie_recommender.get_recommendations(preference, top_k=5)
    
    if not recommendations:
        return "I couldn't find any movies matching your preference. Try asking about a specific genre or actor!"
    
    response = f"🎬 **Movie Recommendations for '{preference}':**\n\n"
    for i, movie in enumerate(recommendations, 1):
        response += f"{i}. **{movie['movie_name']}** ({movie['year']}) - ⭐ {movie['rating']}/10\n"
        response += f"   Genre: {movie['genre']}\n\n"
    
    return response

def calculate_session_duration(session: Dict) -> str:
    """Calculate session duration"""
    try:
        start_time = datetime.fromisoformat(session['start_time'])
        duration = datetime.now() - start_time
        minutes = int(duration.total_seconds() / 60)
        return f"{minutes} minutes"
    except:
        return "Unknown"

def export_conversation(session: Dict) -> str:
    """Export conversation to text"""
    try:
        filename = f"chat_export_{session['session_id']}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Movie Chatbot Conversation Export\n")
            f.write(f"Session ID: {session['session_id']}\n")
            f.write(f"User: {session.get('name', 'Anonymous')}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for turn in session['chat_history']:
                role = "You" if turn['role'] == 'user' else "Assistant"
                f.write(f"{role}: {turn['content']}\n\n")
        
        return f"✅ Conversation exported to `{filename}`"
    except Exception as e:
        return f"❌ Error exporting conversation: {str(e)}"

# 🔷 Enhanced Gradio callbacks
def on_submit(user_message: str, chat_history: List, state: Dict) -> tuple[str, List, Dict]:
    """Enhanced submit handler with better error handling"""
    try:
        response, state = handle_input(user_message, state)
        chat_history.append([user_message, response])
        return "", chat_history, state
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        chat_history.append([user_message, error_msg])
        return "", chat_history, state

def on_clear() -> tuple[List, str, Dict]:
    """Clear chat and reset session"""
    return [], "", session_manager.init_session()

def on_export(state: Dict) -> str:
    """Export current conversation"""
    return export_conversation(state)

# 🔷 Enhanced Gradio app with better UI
with gr.Blocks(theme=gr.themes.Soft(), title="Movie Expert Chatbot 🎬") as demo:
    gr.Markdown("""
    # 🎬 Movie Expert Chatbot
    
    Your intelligent assistant for all things Hollywood movies! Ask about cast, directors, ratings, and get personalized recommendations.
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                container=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask about movies, get recommendations, or type /help for commands...",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary")
                export_btn = gr.Button("Export Chat", variant="secondary")
                help_btn = gr.Button("Help", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Quick Tips")
            gr.Markdown("""
            **Try asking:**
            - "Who directed Titanic?"
            - "What movies feature Leonardo DiCaprio?"
            - "Show me action movies from 1994"
            - "Recommend sci-fi movies"
            
            **Commands:**
            - `/help` - Show all commands
            - `/recommend action` - Get recommendations
            - `/top rated` - Best movies
            - `/stats` - Session info
            """)
    
    state = gr.State(session_manager.init_session())
    
    # Event handlers
    msg.submit(on_submit, [msg, chatbot, state], [msg, chatbot, state])
    submit_btn.click(on_submit, [msg, chatbot, state], [msg, chatbot, state])
    clear_btn.click(on_clear, [], [chatbot, msg, state])
    export_btn.click(on_export, [state], [msg])
    help_btn.click(lambda: ("/help", [], session_manager.init_session()), [], [msg, chatbot, state])

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False, debug=True)
