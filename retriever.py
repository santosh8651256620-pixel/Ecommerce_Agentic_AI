from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    collection_name="products",
    persist_directory=str(DB_PATH),
    embedding_function=embeddings
)

def search_products(query, category=None, max_price=None, k=5):

    filters = []

    if category:
        filters.append({"category": {"$eq": category}})

    if max_price is not None:
        filters.append({"price": {"$lte": float(max_price)}})

    search_filter = None

    if len(filters) == 1:
        search_filter = filters[0]
    elif len(filters) > 1:
        search_filter = {"$and": filters}

    results = vectorstore.similarity_search(
        query,
        k=k,
        filter=search_filter
    )

    return results


if __name__ == "__main__":
    products = search_products(
        query="comfortable running shoes",
        category="Shoes",
        max_price=5000
    )

    for product in products:
        print(product.page_content)
        print("SKU:", product.metadata["sku"])
        print("Price:", product.metadata["price"])
        print("--------------------")