from pathlib import Path

import chromadb
from agents import (
    Agent,
    function_tool,
)

chroma_path = Path(__file__).parent.parent / "chroma"
chroma_client = chromadb.PersistentClient(path=str(chroma_path))
nutrition_db = chroma_client.get_collection(name="nutrition_db")


@function_tool
def calorie_lookup_tool(query: str, max_results: int = 3) -> str:
    """
    Tool function for a RAG database to look up calorie information for specific food items, but not for meals.

    Args:
        query: The food item to look up.
        max_results: The maximum number of results to return.

    Returns:
        A string containing the nutrition information.
    """

    results = nutrition_db.query(query_texts=[query], n_results=max_results)

    if not results["documents"][0]:
        return f"No nutrition information found for: {query}"

    # Format results for the agent
    formatted_results = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        food_item = metadata["food_item"].title()
        calories = metadata["calories_per_100g"]
        category = metadata["food_category"].title()

        formatted_results.append(
            f"{food_item} ({category}): {calories} calories per 100g"
        )

    return "Nutrition Information:\n" + "\n".join(formatted_results)


nutrition_agent = Agent(
    name="Nutrition Assistant",
    instructions="""
    You are a helpful nutrition assistant giving out calorie information.
    You give concise answers.
    If you need to look up calorie information, use the calorie_lookup_tool.
    """,
    tools=[calorie_lookup_tool],
)
