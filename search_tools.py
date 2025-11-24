import os
import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """
    Searches the web for the given query using Tavily.
    Useful for finding current events, news, or specific information not in the model's training data.
    """
    # Check for credentials
    if not os.environ.get("TAVILY_API_KEY"):
        logging.warning("Web Search Tool: TAVILY_API_KEY not found in environment.")
        return "Error: TAVILY_API_KEY not found in .env. Please set it up to use web search."
    
    try:
        search = TavilySearchResults(max_results=5) # Limit to 5 results for brevity
        results = search.invoke({"query": query})
        
        # Format results for the LLM to easily pick up sources
        formatted_results = []
        for i, r in enumerate(results):
            formatted_results.append(f"Source {i+1}: {r['url']}\nContent: {r['content'][:300]}...") # Increased snippet size
        
        if formatted_results:
            return "Search Results:\n" + "\n\n".join(formatted_results)
        else:
            return "No relevant search results found."

    except Exception as e:
        logging.error(f"Tavily Search Tool Error: {e}", exc_info=True)
        return f"Error performing search: {e}"
