run:
	poetry run python src/main.py

backend:
	fastapi dev src/evaluator_api/main.py

frontend:
	streamlit run src/frontend/main.py

setup-nltk:
	poetry run python -c "import nltk; nltk.download('all');"
