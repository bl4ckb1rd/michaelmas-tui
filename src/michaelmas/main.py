import logging
from michaelmas.ui.tui import TuiApp

# Configure root logging to INFO to avoid too much noise from Textual/system
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Explicitly enable DEBUG logging for key libraries and our app
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langgraph").setLevel(logging.DEBUG)
logging.getLogger("langchain_core").setLevel(logging.DEBUG)
logging.getLogger("langchain_google_genai").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG) # See raw API requests

# Enable DEBUG for our application modules
logging.getLogger("tui").setLevel(logging.DEBUG)
logging.getLogger("agent").setLevel(logging.DEBUG)

def main():
    """Launches the Textual TUI application."""
    logging.info("Application starting...")
    try:
        app = TuiApp()
        app.run()
    except Exception as e:
        logging.critical(f"Application crashed: {e}", exc_info=True)
        raise
    logging.info("Application finished.")

if __name__ == "__main__":
    main()
