# @Author: Rachith Aiyappa
# @Date: 2025-05-03

import pytest
import pandas as pd
import numpy as np
import pickle as pkl
from pathlib import Path
from stance_detector.aft_decoding import AFTDecoder

@pytest.fixture
def sample_data(tmp_path):
    # Create sample context and context-free DataFrames
    df_context = pd.DataFrame({
        "ID": [1],
        "Target": ["A"],
        "Tweet": ["tweet1"],
        "label": ["3f--1"],
        "Stance": ["FAVOR"],
        "Prompt": ["prompt"],
        "Prompt_free": ["prompt_free"],
        "class_tokens": [["yes", "no"]],
        "class_token_log_probs": [[{"yes": -0.1, "no": -2.0}]],
        "generated_token": ["yes"],
        "generated_overall_log_prob": [-0.1],
    })
    df_context_free = pd.DataFrame({
        "ID": [1],
        "Target": ["A"],
        "Tweet": ["tweet1"],
        "label": ["3f_free--1"],
        "Stance": ["FAVOR"],
        "Prompt": ["prompt"],
        "Prompt_free": ["prompt_free"],
        "class_tokens": [["yes", "no"]],
        "class_token_log_probs": [[{"yes": -0.3, "no": -1.5}]],
        "generated_token": ["yes"],
        "generated_overall_log_prob": [-0.3],
    })
    context_path = tmp_path / "prompt3f_instruction1-context.pkl"
    context_free_path = tmp_path / "prompt3f_instruction1-context_free.pkl"
    with open(context_path, "wb") as f:
        pkl.dump(df_context, f)
    with open(context_free_path, "wb") as f:
        pkl.dump(df_context_free, f)
    return context_path, context_free_path, df_context, df_context_free

def test_filter_nones():
    data = [{"a": 1, "b": None, "c": np.nan}, {"a": None, "b": 2}]
    AFTDecoder.filter_nones(data)
    assert data == [{"a": 1}, {"b": 2}]

def test_readd():
    data = [{"a": 1.0, "b": 2.0}, {"c": 3.0}]
    result = AFTDecoder.readd(data)
    assert result == [3.0, 3.0]

def test_log_to_prob():
    assert np.isclose(AFTDecoder.log_to_prob(0), 0)
    assert np.isclose(AFTDecoder.log_to_prob(np.log(0.5)), 0.5)
    assert np.isclose(AFTDecoder.log_to_prob(np.log(2)), 2)

def test_decode_pipeline(sample_data, tmp_path):
    context_path, context_free_path, _, _ = sample_data
    decoder = AFTDecoder()
    output_path = tmp_path / "output.pkl"
    result_df = decoder.decode(
        data=context_path,
        context_free_data=context_free_path,
        output_path=output_path
    )
    assert isinstance(result_df, pd.DataFrame)
    assert "aft_label" in result_df.columns
    assert "aft_class_token_probs" in result_df.columns
    # Check that the output file was written
    assert output_path.exists()