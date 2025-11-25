import pytest
import os
from michaelmas.tools.search import web_search

def test_web_search_no_api_key(mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    result = web_search.invoke("query")
    assert "Error: TAVILY_API_KEY not found" in result

def test_web_search_success(mocker):
    mocker.patch.dict(os.environ, {"TAVILY_API_KEY": "test"}, clear=True)
    mock_tavily = mocker.patch("michaelmas.tools.search.TavilySearchResults")
    mock_instance = mock_tavily.return_value
    
    mock_results = [
        {"url": "http://example.com", "content": "Example content"}
    ]
    mock_instance.invoke.return_value = mock_results
    
    result = web_search.invoke("query")
    
    assert "Search Results:" in result
    assert "http://example.com" in result
    assert "Example content" in result

def test_web_search_no_results(mocker):
    mocker.patch.dict(os.environ, {"TAVILY_API_KEY": "test"}, clear=True)
    mock_tavily = mocker.patch("michaelmas.tools.search.TavilySearchResults")
    mock_instance = mock_tavily.return_value
    mock_instance.invoke.return_value = []
    
    result = web_search.invoke("query")
    
    assert "No relevant search results found" in result

def test_web_search_exception(mocker):
    mocker.patch.dict(os.environ, {"TAVILY_API_KEY": "test"}, clear=True)
    mock_tavily = mocker.patch("michaelmas.tools.search.TavilySearchResults")
    mock_instance = mock_tavily.return_value
    mock_instance.invoke.side_effect = Exception("Boom")
    
    result = web_search.invoke("query")
    
    assert "Error performing search: Boom" in result
