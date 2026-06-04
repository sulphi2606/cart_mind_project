#---------------Dependencies----------------
# pip install datasets pandas pymongo python-dotenv tqdm

import os
import ast
from dotenv import load_dotenv
from datasets import load_dataset
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm

# ════════════════════════════════════════
# loading api credentials
# ════════════════════════════════════════
load_dotenv(".env")

MONGO_URI = os.getenv("MONGO_URI")
os.environ.pop("MONGO_URI", None)   # prevent chromadb conflict

if not MONGO_URI:
    raise ValueError("❌ MONGO_URI missing in .env")

print("✅ Credentials loaded")

# ════════════════════════════════════════
# Config
# ════════════════════════════════════════
DB_NAME         = "amazon_rag_db"
COLLECTION_NAME = "products"
MAX_RECORDS     = 1000   # change to None for all 204k

# ════════════════════════════════════════
# Step 1 — Load from HuggingFace
# ════════════════════════════════════════
print("📥 Loading dataset from HuggingFace...")
dataset = load_dataset(
    "minhth2nh/amazon_product_review_283K",
    split="train"
)
df = dataset.to_pandas()
print(f"✅ Loaded {len(df)} rows")

if MAX_RECORDS:
    df = df.head(MAX_RECORDS)
    print(f"⚡ Using first {MAX_RECORDS} records")

# ════════════════════════════════════════
# Step 2 — Clean helper functions
# ════════════════════════════════════════
def safe_parse_list(val):
    if not val or val in ["[]", "", None]:
        return ""
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return ". ".join([str(i) for i in parsed if i])
    except Exception:
        pass
    return str(val)

def clean_price(val):
    try:
        return str(round(float(str(val).replace("$", "").replace(",", "").strip()), 2))
    except Exception:
        return ""

def build_embed_text(row):
    parts = [
        row["product_name"],
        row["description"][:300] if row["description"] else "",
        row["features"][:200]    if row["features"]    else "",
        f"Category: {row['category']}",
        f"Brand: {row['brand']}",
    ]
    return " | ".join([p for p in parts if p.strip()])

# ════════════════════════════════════════
# Step 3 — Clean columns
# ════════════════════════════════════════
print("🧹 Cleaning data...")

df["product_name"]    = df["title_y"].fillna("").str.strip()
df["description_str"] = df["description"].apply(safe_parse_list)
df["features_str"]    = df["features"].apply(safe_parse_list)
df["price_clean"]     = df["price"].apply(clean_price)
df["avg_rating"]      = pd.to_numeric(df["average_rating"], errors="coerce")
df["category"]        = df["main_category"].fillna("General").str.strip()
df["brand"]           = df["store"].fillna("Unknown").str.strip()
df["review_text"]     = df["text"].fillna("").str.strip()

df = df[df["product_name"].str.len() > 3]

# ════════════════════════════════════════
# Step 4 — Deduplicate by ASIN
# ════════════════════════════════════════
print("🔄 Deduplicating by ASIN...")
product_df = df.groupby("asin").agg(
    product_name = ("product_name",   "first"),
    description  = ("description_str","first"),
    features     = ("features_str",   "first"),
    price        = ("price_clean",    "first"),
    avg_rating   = ("avg_rating",     "first"),
    rating_count = ("review_text",    "count"),
    category     = ("category",       "first"),
    brand        = ("brand",          "first"),
    all_reviews  = ("review_text", lambda x: " | ".join(x.dropna().head(5)))
).reset_index()

print(f"✅ Unique products: {len(product_df)}")

# ════════════════════════════════════════
# Step 5 — Build embed_text
# ════════════════════════════════════════
product_df["embed_text"] = product_df.apply(build_embed_text, axis=1)

# ════════════════════════════════════════
# Step 6 — Push to MongoDB
# ════════════════════════════════════════
print("🍃 Connecting to MongoDB...")
client = MongoClient(MONGO_URI)
col    = client[DB_NAME][COLLECTION_NAME]

col.delete_many({})
print("🗑️  Cleared existing collection")

docs = []
for _, row in product_df.iterrows():
    # use reviews as description fallback
    description = row["description"]
    if not description or description.strip() in ["", "[]"]:
        description = row["all_reviews"][:300]

    docs.append({
        "asin":         row["asin"],
        "product_name": row["product_name"],
        "description":  description,
        "features":     row["features"],
        "price":        row["price"],
        "avg_rating":   float(row["avg_rating"]) if pd.notna(row["avg_rating"]) else None,
        "rating_count": int(row["rating_count"]),
        "category":     row["category"],
        "brand":        row["brand"],
        "all_reviews":  row["all_reviews"],
        "embed_text":   row["embed_text"],
    })

for i in tqdm(range(0, len(docs), 500), desc="📤 Inserting"):
    col.insert_many(docs[i : i + 500])

print(f"\n🎉 Done! {col.count_documents({})} products in MongoDB")
client.close()
