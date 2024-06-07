run:
	poetry run python src/main.py

setup-nltk:
	poetry run python -c "import nltk; nltk.download('all');"
