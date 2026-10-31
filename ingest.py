import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "data" / "catalog.csv"
DB_PATH = BASE_DIR / "chroma_db"

def ingest_products():
    # Read product data
    df = pd.read_csv(CSV_PATH)

    documents = []

    for _, row in df.iterrows():
        text = f"""
        Product: {row['name']}
        Category: {row['category']}
        Price: {row['price']}
        Description: {row['description']}
        Features: {row['features']}
        """

        metadata = {
            "sku": str(row["sku"]),
            "name": str(row["name"]),
            "category": str(row["category"]),
            "price": float(row["price"]),
            "in_stock": str(row["in_stock"]),
            "stock_qty": int(row["stock_qty"])
        }

        documents.append(
            Document(page_content=text, metadata=metadata)
        )

    # Create embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Save products in Chroma database
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="products",
        persist_directory=str(DB_PATH)
    )

    print(f"Successfully added {len(documents)} products!")

if __name__ == "__main__":
    ingest_products()