# Emotion Evaluator App

This project contains an emotion evaluator app. You can give some text as input and the evaluator will tell you if the sentiment of that text is positive or negative.

![Screenshot from 2024-06-10 19-08-42](https://github.com/RobinDeSmet/EmotionEvaluator/assets/36922800/382f2af3-e0f1-4889-a49b-844e9ba46caa)

Accompanied with this app is a benchmark cli tool that let's you benchmark pretrained sentiment analysis models on a dataset. For example, a dataset filled with movie reviews from IMDB.
The benchmark will contain following items:

- The preprocessed text and the predicted sentiment
- A classification report
- A confusion matrix
- ROC curve

## Setup

### 1. Docker

If you want to use the containarized version of the app you will need Docker Desktop. You can install it via the [official documentation](https://www.docker.com/products/docker-desktop/).

- Navigate to the root directory of the project
- Run `docker compose up -d`

### 2. Development environment

This project uses poetry, to install it you can follow this [guide](https://python-poetry.org/docs/#installing-with-the-official-installer).

- Navigate to the root directory of the project
- Create a copy of the `.env.template` file and rename it to: `.env`
- Run `poetry install`
- Run `poetry shell`
- To start the backend run: `make backend`. This will run a Fast API server with the endpoints needed to run the Emotion Evaluator.
- In another window run `make frontend`. This will start a streamlit frontend that will serve the Emotion Evaluator UI.

### 3. Benchmark CLI tool

By default, the Emotion Evaluator uses, [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), as its pretrained sentiment analysis model.

However, you can play around and create benchmarks for other pretrained models as well. The benchmark tool works for models developed in both PyTorch and TensorFlow. The tool can handle 5 star based labels or descriptive labels ('Positive', 'Negative', 'Neutral',...). Here is how you can setup and use the CLI tool:

- Navigate to the root directory of the project
- Create a copy of the `.env.template` file and rename it to: `.env`
- Run `poetry install`
- Run `poetry shell`
- `poetry run python src/main.py`: This will run the benchmark for the default model used by the Emotion Evaluator. You can specify following parameters in the CLI tool:
  - `output_dir (defaults to "src/results")`: This will set the output directory, where the results of the benchmark will be stored.
  - `model (defaults to "distilbert/distilbert-base-uncased-finetuned-sst-2-english")`: The pretrained model that you wish to benchmark.
  - `data (defaults to "src/data/IMDB-movie-reviews.csv"`: The path to the CSV file that holds the dataset for the benchmark.
  - `sequence_length (defaults to 512)`: The maximum sequence length possible for the model that you want to test.
  - `framework (defaults to "tf")`: Tells the benchmark script which framework to use (tf or pt).

## Some other useful commands:

- `make test`: Run the test suite for this project
- `make lint`: Run pylint to make sure that your code respects the PEP8 guidelines.
- `make setup-nltk`: If you decide to use some of the features of nltk to experiment with certain preprocessing steps, don't forget to run this command.
  poetry run python -c "import nltk; nltk.download('all');"

Enjoy!
