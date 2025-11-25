.PHONY: install run clean

VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python

install: $(VENV_DIR)
	@echo "Installing dependencies..."
	@pip install uv || echo "uv already installed or could not install globally. Assuming it's available."
	@$(VENV_DIR)/bin/uv pip install -e .

$(VENV_DIR):
	@echo "Creating virtual environment..."
	@python3 -m venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)"

run: $(VENV_DIR)
	@echo "Running Michaelmas app..."
	@PYTHONPATH=src uv run -m michaelmas.main

test: $(VENV_DIR)
	@echo "Running tests..."
	@PYTHONPATH=src uv run pytest --cov=michaelmas --cov-report=term-missing tests/

clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV_DIR)
	@rm -f app.log
	@rm -rf .conversations
	@rm -f token.json
	@rm -f uv.lock
	@find . -name "__pycache__" -type d -exec rm -r {} +
	@find . -name "*.pyc" -exec rm {} +
	@echo "Clean up complete."
