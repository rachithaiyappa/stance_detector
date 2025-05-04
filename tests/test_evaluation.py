# @Author: Rachith Aiyappa
# @Date: 2025-05-02

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from stance_detector.evaluation import Evaluator
import pickle as pkl

@pytest.fixture
def sample_eval_data(tmp_path):
    # Create a sample DataFrame for evaluation
    df = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "Stance": ["FAVOR", "AGAINST", "FAVOR", "AGAINST"],
        "generated_token": ["true", "false", "true", "false"]
    })
    pkl_path = tmp_path / "prompt3f_instruction1.pkl"
    with open(pkl_path, "wb") as f:
        pkl.dump(df, f)
    return pkl_path, df

def test_mapping_favor():
    assert Evaluator.mapping_favor("true") == "FAVOR"
    assert Evaluator.mapping_favor("false") == "AGAINST"
    assert Evaluator.mapping_favor("other") == "NONE"

def test_mapping_against():
    assert Evaluator.mapping_against("true") == "AGAINST"
    assert Evaluator.mapping_against("false") == "FAVOR"
    assert Evaluator.mapping_against("other") == "NONE"

def test_mapping_positive_negative():
    assert Evaluator.mapping_positive_negative("positive") == "FAVOR"
    assert Evaluator.mapping_positive_negative("negative") == "AGAINST"
    assert Evaluator.mapping_positive_negative("neutral") == "NONE"

def test_mapping_favor_against():
    assert Evaluator.mapping_favor_against("favor") == "FAVOR"
    assert Evaluator.mapping_favor_against("against") == "AGAINST"
    assert Evaluator.mapping_favor_against("none") == "NONE"

def test_evaluation_metrics():
    df = pd.DataFrame({
        "Stance": ["FAVOR", "FAVOR", "AGAINST", "AGAINST"],
        "Assigned": ["FAVOR", "AGAINST", "AGAINST", "FAVOR"]
    })
    tp, fp, tn, fn = Evaluator.evaluation_against(df)
    assert tp == 1
    assert fp == 1
    assert tn == 1
    assert fn == 1

    tp_f, fp_f, tn_f, fn_f = Evaluator.evaluation_favor(df)
    assert tp_f == 1
    assert fp_f == 1
    assert tn_f == 1
    assert fn_f == 1

def test_evaluate_pipeline(sample_eval_data, tmp_path):
    pkl_path, _ = sample_eval_data
    evaluator = Evaluator()
    # Should not raise and should return a dict with expected keys
    result = evaluator.evaluate(
        input_path=pkl_path,
        prompt_template="3f",
        instruction="1",
        output_path=tmp_path / "eval_output.csv"
    )
    assert isinstance(result, dict)
    assert "favor_f1" in result
    assert "against_f1" in result
    assert "overall_f1" in result