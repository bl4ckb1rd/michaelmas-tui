.PHONY: install run test clean

install:
	@echo "Installing dependencies with uv..."
	@uv sync
	@echo "Installing pre-commit hooks..."
	@uv run pre-commit install

run:
	@echo "Running Michaelmas app..."
	@PYTHONPATH=src uv run python -m michaelmas.main

test:
	@echo "Running tests..."
	@PYTHONPATH=src uv run pytest --cov=src tests/

clean:
	@echo "Cleaning up..."
	@rm -rf .venv
	@rm -f app.log
	@rm -rf .conversations
	@rm -f token.json
	@find . -name "__pycache__" -type d -exec rm -r {} +
	@find . -name "*.pyc" -exec rm {} +
	@echo "Clean up complete."