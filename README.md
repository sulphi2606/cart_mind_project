# 🧠 CartMind AI — Smart Product Recommendation Agent

> **Think Less. Shop Smarter.**  
> An AI-powered product recommendation chatbot built on Amazon's 283K product dataset.

---

## 🚀 What is CartMind?

CartMind is a conversational AI shopping assistant that helps users find the perfect Amazon product using natural language. Just describe what you need — CartMind finds the best match from a curated product database using semantic search and generative AI.

---

## ✨ Features

- 🔍 **Semantic Search** — understands natural language queries, not just keywords
- 🤖 **AI Recommendations** — powered by Google Gemini for intelligent, context-aware responses
- 🧠 **Multi-turn Memory** — remembers the last 4 exchanges for follow-up questions
- 📦 **283K Amazon Products** — jewelry, fashion, accessories, and more
- ⚡ **Fast Retrieval** — Chroma vector database for sub-second similarity search
- 💬 **Clean Chat UI** — built with Streamlit for a smooth experience

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.0 Flash |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Store | Chroma DB |
| Database | MongoDB Atlas |
| Dataset | Amazon Product Review 283K (HuggingFace) |
| Framework | LangChain LCEL |
| Frontend | Streamlit |

---

## 📁 Project Structure

```
cart_mind_project/
├── agents.py          # RAG pipeline, LangChain chains, initialize()
├── streamlit_app.py   # Streamlit frontend UI
├── mongo.py           # HuggingFace → MongoDB ingestion script
├── requirements.txt   # Python dependencies
├── .gitignore         # Excludes .env, chroma_db, __pycache__
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/cart_mind_project.git
cd cart_mind_project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_gemini_api_key
MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net/
```

- Get Gemini API key: [aistudio.google.com](https://aistudio.google.com)
- Get MongoDB URI: [cloud.mongodb.com](https://cloud.mongodb.com)

### 4. Load the dataset into MongoDB
```bash
python mongo.py
```
This downloads the Amazon dataset from HuggingFace, cleans it, and pushes 1000 products to MongoDB.

### 5. Run the app
```bash
streamlit run streamlit_app.py
```

---

## 💡 Example Queries

- *"Best watch under $20"*
- *"Gift necklace for mom"*
- *"Top rated fashion accessories"*
- *"Lightweight bag for travel"*
- *"Something unique and trendy under $30"*

---

## 🌐 Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Under **Advanced Settings → Secrets**, add:
```toml
GEMINI_API_KEY = "your_gemini_api_key"
MONGO_URI = "mongodb+srv://user:password@cluster.mongodb.net/"
```
5. Click **Deploy**

---

## 📊 How It Works

```
User Query
    ↓
Embed query (BAAI/bge-small-en-v1.5)
    ↓
Similarity Search (Chroma DB, k=5)
    ↓
Retrieve top 5 products from MongoDB
    ↓
Format context + chat history
    ↓
Gemini 2.0 Flash generates recommendation
    ↓
Response displayed in Streamlit UI
```

---

## 🔒 Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
| `MONGO_URI` | MongoDB Atlas connection string |

> ⚠️ Never commit your `.env` file. It is excluded via `.gitignore`.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

*Built with ❤️ using LangChain, Streamlit, and Google Gemini*
