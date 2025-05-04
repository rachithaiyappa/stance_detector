# @Author: Rachith Aiyappa
# @Date: 2025-05-02

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from stance_detector.pmi_decoding import PMIDecoder

@pytest.fixture
def sample_data(tmp_path):
    # Create sample context and context-free DataFrames
    df_context = pd.DataFrame({
        "ID": [1, 2],
        "Target": ["A", "B"],
        "Tweet": ["tweet1", "tweet2"],
        "label": ["3f--1", "3f--1"],
        "Stance": ["FAVOR", "AGAINST"],
        "Prompt": ["prompt", "prompt"],
        "class_tokens": [["yes"], ["no"]],
        "class_token_log_probs": [[{"yes": -0.1}], [{"no": -0.2}]],
        "generated_token": ["yes", "no"],
        "generated_overall_log_prob": [-0.1, -0.2],
    })
    df_context_free = pd.DataFrame({
        "ID": [1, 2],
        "Target": ["A", "B"],
        "Tweet": ["tweet1", "tweet2"],
        "label": ["3f_free--1", "3f_free--1"],
        "Stance": ["FAVOR", "AGAINST"],
        "Prompt": ["prompt_free", "prompt_free"],
        "class_tokens": [["yes"], ["no"]],
        "class_token_log_probs": [[{"yes": -0.3}], [{"no": -0.4}]],
        "generated_token": ["yes", "no"],
        "generated_overall_log_prob": [-0.3, -0.4],
    })
    context_path = tmp_path / "prompt3f_instruction1-context.parquet"
    context_free_path = tmp_path / "prompt3f_instruction1-context_free.parquet"
    df_context.to_parquet(context_path, index=False)
    df_context_free.to_parquet(context_free_path, index=False)
    return context_path, context_free_path, df_context, df_context_free

def test_filter_nones():
    data = [{"a": 1, "b": None, "c": np.nan}, {"a": None, "b": 2}]
    PMIDecoder.filter_nones(data)
    assert data == [{"a": 1}, {"b": 2}]

def test_readd():
    data = [{"a": 1.0, "b": 2.0}, {"c": 3.0}]
    result = PMIDecoder.readd(data)
    assert result == 6.0

def test_explode_and_merge():
    df = pd.DataFrame({
        "ID": [1],
        "class_tokens": [["a", "b"]],
        "class_token_log_probs_total": [[0.1, 0.2]]
    })
    result = PMIDecoder.explode_and_merge(df, ["class_tokens", "class_token_log_probs_total"])
    assert len(result) == 2
    assert set(result["class_tokens"]) == {"a", "b"}

def test_deduplicate_context_pmi():
    df = pd.DataFrame({
        "ID": [1, 1, 2],
        "val": [10, 20, 30]
    })
    dedup = PMIDecoder.deduplicate_context_pmi(df)
    assert dedup["ID"].tolist() == [1, 2]

def test_decode_pipeline(sample_data, tmp_path):
    context_path, context_free_path, _, _ = sample_data
    decoder = PMIDecoder()
    # Should not raise and should return a DataFrame
    result = decoder.decode(
        data=context_path,
        context_free_data=context_free_path,
        output_path=tmp_path / "output.parquet"
    )
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "pmi_lognum-logden" in result.columns