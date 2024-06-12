"""Test for the preprocessor"""

from src.preprocessor import Preprocessor

preprocessor = Preprocessor(sequence_length=32)


def test_evaluate_read_data_works():
    """Test if the read in data function works correctly."""
    # Read in the data
    data = preprocessor.read_in_data()

    # Check the result
    assert len(data) == 100
    assert data.columns.tolist() == ["review", "sentiment"]


def test_evaluate_preprocess_text_works():
    """Test if the preprocess text function works correctly."""
    test_data = "<br></br>Such a @myself    greta and beautifil movie. https://malicous_url.com <div> I loved the acting.</div>"

    # Preprocess the text
    resulting_text = preprocessor.preprocess_text(test_data, use_autocorreect=True)

    print(resulting_text)

    # Check the result
    assert resulting_text[0] == "Such a great and beautiful"
    assert resulting_text[1] == "movie. I loved the acting."
