#---------------Dependencies----------------
# pip install langchain langchain-google-genai python-dotenv
# pip install langchain_community sentence-transformers
# pip install langchain-chroma chromadb pymongo numpy

import os
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma


# ════════════════════════════════════════
# STEP 1 — loading api credentials
# ════════════════════════════════════════
load_dotenv(".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI      = os.getenv("MONGO_URI")

# remove from env to prevent chromadb pydantic conflict
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("MONGO_URI", None)

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY missing in .env")
if not MONGO_URI:
    raise ValueError("❌ MONGO_URI missing in .env")

print("API credentials loaded")
print("GEMINI :", GEMINI_API_KEY[:10], "...")
print("MONGO  :", MONGO_URI[:30],      "...")


# ════════════════════════════════════════
# STEP 2 — model initialization
# ════════════════════════════════════════
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,
    max_tokens=1024
)
print("Model loaded")


# ════════════════════════════════════════
# STEP 4 — binding llm with dynamic prompt template
# (STEP 3 test call moved to __main__)
# ════════════════════════════════════════
prompts = ChatPromptTemplate(
    [
        ("system", """You are CartMind, a helpful Amazon product
recommendation assistant. Use ONLY the product context provided
to answer. For each product ALWAYS mention:
  - Product name
  - Price (if not available say "Check Amazon for price")
  - Rating
  - Why it suits the user's need
  - Brand name
Be specific. If something partially matches, recommend it anyway."""),
        ("human", """
Product Context:
{context}

Previous Conversation:
{history}

User Question: {question}

Your Recommendation:""")
    ]
)

# binding llm with dynamic prompt template
chain = prompts | model | StrOutputParser()

print("Prompt template created")
print("LLM bound with prompt using LCEL ( prompts | model )")


# ════════════════════════════════════════
# Global variables
# (set inside initialize(), used by Streamlit)
# ════════════════════════════════════════
retriever   = None
format_docs = None
rag_chain   = None


# ════════════════════════════════════════
# initialize() — called by Streamlit once
# wraps Steps 5-11 so they don't run on import
# ════════════════════════════════════════
def initialize():
    global retriever, format_docs, rag_chain

    # ════════════════════════════════════════
    # STEP 5 — Document loading from MongoDB
    # ════════════════════════════════════════
    print("🍃 Connecting to MongoDB...")
    client     = MongoClient(MONGO_URI)
    collection = client["amazon_rag_db"]["products"]

    raw_products = list(collection.find(
        {},
        {
            "asin": 1, "product_name": 1, "embed_text": 1,
            "description": 1, "price": 1, "avg_rating": 1,
            "category": 1, "brand": 1, "all_reviews": 1,
            "_id": 0
        }
    ))
    client.close()

    # convert each product → LangChain Document
    raw_docs = []
    for p in raw_products:
        # clean price
        price = p.get("price", "")
        try:
            price = f"${float(price):.2f}"
        except:
            price = "Check Amazon for price"

        # clean rating
        rating = p.get("avg_rating", "")
        try:
            rating = f"{float(rating):.1f} ⭐"
        except:
            rating = "Not rated"

        # use reviews as description fallback
        description = p.get("description", "")
        if not description or description.strip() in ["", "[]"]:
            description = p.get("all_reviews", "")[:300]

        doc = Document(
            page_content=p.get("embed_text", p.get("product_name", "")),
            metadata={
                "asin":         p.get("asin", ""),
                "product_name": p.get("product_name", ""),
                "price":        price,
                "avg_rating":   rating,
                "category":     p.get("category", ""),
                "brand":        p.get("brand", ""),
                "description":  description,
                "all_reviews":  p.get("all_reviews", "")[:200],
            }
        )
        raw_docs.append(doc)

    print("Before splitting --", len(raw_docs))

    # ════════════════════════════════════════
    # STEP 6 — Document Splitting
    # ════════════════════════════════════════
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    splitted_docs = text_splitter.split_documents(raw_docs)
    print("After splitting ---", len(splitted_docs))

    # ════════════════════════════════════════
    # STEP 7 — Embedding
    # ════════════════════════════════════════
    embedding = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # ════════════════════════════════════════
    # STEP 8 & 9 — Chroma vector store
    # ════════════════════════════════════════
    persist_dir = "chroma_db"

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("Loading existing Chroma DB from disk...")
        vectorsdb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding
        )
    else:
        print("Creating Chroma DB — embedding (3-5 mins)...")
        vectorsdb = Chroma.from_documents(
            splitted_docs,
            embedding,
            persist_directory=persist_dir
        )

    print("Chroma DB ready:", vectorsdb._collection.count(), "vectors")

    # ════════════════════════════════════════
    # STEP 11 — Full RAG chain
    # ════════════════════════════════════════
    retriever = vectorsdb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    def _format_docs(docs):
        formatted = []
        for doc in docs:
            m = doc.metadata
            formatted.append(
                f"Product : {m.get('product_name', 'N/A')}\n"
                f"Price   : {m.get('price', 'N/A')}\n"
                f"Rating  : {m.get('avg_rating', 'N/A')}\n"
                f"Category: {m.get('category', 'N/A')}\n"
                f"Brand   : {m.get('brand', 'N/A')}\n"
                f"Info    : {m.get('description', 'N/A')}"
            )
        return "\n\n---\n\n".join(formatted)

    format_docs = _format_docs

    rag_chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
            "history":  lambda _: ""
        }
        | prompts
        | model
        | StrOutputParser()
    )

    print("✅ CartMind RAG chain ready!")


# ════════════════════════════════════════
# STEP 12 — Dynamic query with multi-turn memory
# (used in terminal mode)
# ════════════════════════════════════════
chat_history = []

def ask(question: str) -> str:
    history_text = ""
    if chat_history:
        for human, ai in chat_history[-4:]:
            history_text += f"User: {human}\nAssistant: {ai}\n\n"

    return (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
            "history":  lambda _: history_text
        }
        | prompts
        | model
        | StrOutputParser()
    ).invoke(question)


# ════════════════════════════════════════
# Run in terminal mode
# ════════════════════════════════════════
if __name__ == "__main__":

    # STEP 3 — quick model test (only in terminal mode)
    print("\n--- Model test ---")
    response = model.invoke("say hello in one line")
    print(response.content)
    print(response.usage_metadata)

    # STEP 4 test — quick chain test (only in terminal mode)
    print("\n--- Chain test ---")
    resp = chain.invoke({
        "context":  "No products loaded yet",
        "history":  "No history",
        "question": "hello"
    })
    print(resp)

    # STEP 10 — similarity search test (runs after initialize)
    initialize()

    print("\n--- Similarity search test ---")
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_chroma import Chroma
    _emb = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    _vdb = Chroma(persist_directory="chroma_db", embedding_function=_emb)
    result = _vdb.similarity_search("what is a good gift necklace", k=3)
    print(result)
    print(len(result))

    print("\n" + "="*50)
    print("  🧠 CartMind AI — Product Recommendation Agent")
    print("="*50)
    print("  'history' → see past conversation")
    print("  'clear'   → clear memory")
    print("  'exit'    → quit")
    print("="*50)

    while True:
        user_input = input("\n👤 You: ").strip()

        if not user_input:
            continue
        elif user_input.lower() == "exit":
            print("Goodbye!")
            break
        elif user_input.lower() == "clear":
            chat_history.clear()
            print("Memory cleared!")
            continue
        elif user_input.lower() == "history":
            if not chat_history:
                print("No history yet.")
            else:
                for i, (h, a) in enumerate(chat_history, 1):
                    print(f"[{i}] 👤 {h}")
                    print(f"    🤖 {a[:120]}...")
            continue

        print("\n🔍 Searching products...\n")
        resp = ask(user_input)
        print(f"🤖 CartMind: {resp}")
        print("-" * 50)

        chat_history.append((user_input, resp))
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]