🎬 `movie-rag-chatbot` – Conversational Movie Q\&A with OpenAI + Gradio
The `movie-rag-chatbot` is an intelligent, interactive chatbot that allows users to ask natural language questions about a curated dataset of top Hollywood movies. It combines Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Gradio to deliver accurate and conversational insights based solely on the dataset.

 🚀 Features

* ✅ Chat-based Interface using Gradio
* 🤖 GPT-3.5-Turbo Integration with OpenAI API
* 🔍 RAG-Style Processing — answers based on actual data, not hallucinations
* 📊 Dynamic Movie Knowledge loaded from an Excel file
* 🧾 User Data Capture (name, phone, email, location)
* 📁 Session-aware Chat with persistent history during each session
* 💾 User Info Logging to Excel (`user_info.xlsx`)

 🧠 How It Works (RAG Flow)

1. Data Retrieval: Loads and formats Hollywood movie data from Excel
2. Prompt Engineering: Combines the user query with the movie knowledge
3. LLM Processing: Sends prompt to GPT-3.5 via OpenAI API
4. Response Generation: Returns relevant answers using only known data
5. Session Memory: Maintains chat history and personalizes responses

 🧰 Tech Stack

| Tool       | Purpose                          |
| ---------- | -------------------------------- |
| OpenAI | GPT-3.5-Turbo for Q\&A logic     |
| Gradio | Interactive web UI               |
| Pandas | Data handling and processing     |
| .env   | Secure API key management        |
| Excel  | Both input data and user logging |

 📂 Structure
```bash
movie-rag-chatbot/
├── app/
│   └── movie_chatbot.py        Main chatbot logic
├── data/
│   ├── Hollywood_Top_Movies.xlsx   Movie knowledge base
│   └── user_info.xlsx          Stores collected user info
├── .env.example                Environment variable template
├── requirements.txt            Required Python packages
└── README.md
```

 🛠️ Setup
1. Clone the repository
2. Run `pip install -r requirements.txt`
3. Rename `.env.example` to `.env` and add your OpenAI key
4. Run the chatbot

```bash
python app/movie_chatbot.py
```
Here’s a detailed and professional project description for your `movie-rag-chatbot` GitHub repository:

 ✅ Use Cases

* Ask: *“Who directed Titanic?”*
* Ask: *“Which movies feature Leonardo DiCaprio?”*
* Ask: *“List Action movies from 1994.”*

The model will answer based on your Excel dataset and outside the dataset, making this a reliable domain-specific AI assistant.



