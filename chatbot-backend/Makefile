.PHONY: help setup install test run run-dev clean docker-build docker-run docker-stop format

help:  ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

setup: ## Setup development environment
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	cp .env.example .env
	mkdir -p data/visualizations models logs

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run tests
	pytest tests/ -v

run: ## Run production server
	python run.py --host 0.0.0.0 --port 8000

run-dev: ## Run development server with reload
	python run.py --host 0.0.0.0 --port 8000 --reload --workers 1

clean: ## Clean temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".hypothesis" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/

format: ## Format code with black
	black .
	isort .

docker-build: ## Build Docker image
	docker build -t chatbot-backend .

docker-run: ## Run Docker container
	docker-compose up -d

docker-stop: ## Stop Docker container
	docker-compose down

jupyter: ## Start Jupyter notebook server
	jupyter notebook notebooks/

process-data: ## Run data processing pipeline
	jupyter nbconvert --to notebook --execute notebooks/1_data_preprocessing_cleaning.ipynb --inplace
	jupyter nbconvert --to notebook --execute notebooks/2_exploratory_data_analysis.ipynb --inplace
	jupyter nbconvert --to notebook --execute notebooks/3_data_visualization.ipynb --inplace
	jupyter nbconvert --to notebook --execute notebooks/4_modeling.ipynb --inplace
	jupyter nbconvert --to notebook --execute notebooks/5_langchain_data_preparation.ipynb --inplace

logs: ## View application logs
	tail -f logs/app.log

dependencies: ## List project dependencies
	pip freeze > requirements.txt

update-deps: ## Update project dependencies
	pip install --upgrade -r requirements.txt

lint: ## Run linting checks
	flake8 .
	pylint src/
	mypy src/

init-dev: setup install format lint ## Initialize development environment

# Database commands (for future use)
# db-init: ## Initialize database
# 	python scripts/init_db.py

# db-migrate: ## Run database migrations
# 	alembic upgrade head

# db-rollback: ## Rollback database migration
# 	alembic downgrade -1
