import pandas as pd
from pathlib import Path
from langchain_core.tools import tool
from retriever import search_products

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "data" / "catalog.csv"


@tool
def check_product_availability(sku: str) -> str:
    """Check product price and stock availability using product SKU."""

    df = pd.read_csv(CSV_PATH)
    product = df[df["sku"].astype(str).str.upper() == sku.upper()]

    if product.empty:
        return f"Product with SKU {sku} was not found."

    item = product.iloc[0]

    if str(item["in_stock"]).lower() == "true" and int(item["stock_qty"]) > 0:
        return (
            f"Product: {item['name']}\n"
            f"Price: Rs. {item['price']}\n"
            f"Available stock: {item['stock_qty']}"
        )

    return f"{item['name']} is currently out of stock."

@tool
def search_product_catalog(query: str) -> str:
    """Search products from the product catalog using a natural language query."""

    results = search_products(query=query, k=5)

    if not results:
        return "No matching products found."

    output = []

    for product in results:
        output.append(
            f"Product: {product.metadata['name']}\n"
            f"SKU: {product.metadata['sku']}\n"
            f"Category: {product.metadata['category']}\n"
            f"Price: Rs. {product.metadata['price']}"
        )

    return "\n\n".join(output)
@tool
def search_product_policy_info(query: str) -> str:
    """Search product policy information. Policy search will be connected by the policy team member."""

    return (
        "Product policy search is not connected yet. "
        "Please connect the policy retrieval function from the policy team member."
    )
if __name__ == "__main__":
    result = check_product_availability.invoke({"sku": "P001"})
    print(result)