import os
from dotenv import load_dotenv
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

def list_all_models():
    """
    Uses the google.generativeai library (if available) or direct API calls
    to list all models available to the API key.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        return

    print("Attempting to list models using google-generativeai...")
    
    try:
        # We try to import here because it might have been installed by previous steps
        # or we can try to install it.
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        print("\n--- Available Models ---")
        found_any = False
        for m in genai.list_models():
            found_any = True
            print(f"Name: {m.name}")
            print(f"Supported generation methods: {m.supported_generation_methods}")
            print("-" * 20)
        
        if not found_any:
            print("No models found. This suggests an issue with the API key or project configuration.")
        else:
            print("\nSuccess! Please use one of the 'Name' values above in your agent configuration.")

    except ImportError:
        print("The google-generativeai library is not installed.")
        print("Please run: uv add google-generativeai")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    list_all_models()