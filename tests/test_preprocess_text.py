# @Author: Rachith Aiyappa
# @Date: 2025-05-14

import pandas as pd
import pytest
from pathlib import Path
from stance_detector.preprocess_text import PreProcessText
import tempfile


@pytest.fixture
def sample_tweet_csv(tmp_path):
    """Creates a temporary CSV file with sample tweet data."""
    df = pd.DataFrame(
        {
            "Tweet": [
                "I love clean energy!",
                "Climate change is a hoax.",
                "We need to reduce emissions now.",
            ]
        }
    )
    input_csv = tmp_path / "test_tweets.csv"
    df.to_csv(input_csv, index=False)
    return input_csv


def test_preprocess_text_creates_output_and_column(sample_tweet_csv, tmp_path):
    output_csv = tmp_path / "processed_tweets.csv"
    preprocessor = PreProcessText()

    df = preprocessor.preprocess_text(
        input_path=sample_tweet_csv, output_path=output_csv
    )

    # Check that the output file is created
    assert output_csv.exists(), "Output CSV file was not created"

    # Check that the returned DataFrame has the expected column
    assert (
        "bertweet_preprocessed" in df.columns
    ), "Missing preprocessed column in DataFrame"
    assert all(
        df["bertweet_preprocessed"].apply(lambda x: isinstance(x, str))
    ), "Preprocessed text should be strings"

    # Load saved file and compare
    df_saved = pd.read_csv(output_csv)
    assert df_saved.equals(df), "Saved DataFrame does not match returned DataFrame"
