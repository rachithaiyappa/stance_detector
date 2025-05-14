# @Author: Rachith Aiyappa
# @Date: 2025-05-01

import pytest
import pandas as pd
import os
from stance_detector.flant5_runner import FlanT5Runner


class DummyTokenizer:
    def __init__(self):
        pass

    def __call__(self, text, return_tensors=None):
        import torch

        return type("Dummy", (), {"input_ids": torch.tensor([[1, 2]])})()

    def decode(self, ids, skip_special_tokens=True):
        return "dummy_token"

    def from_pretrained(*args, **kwargs):
        return DummyTokenizer()


class DummyModel:
    def __init__(self):
        pass

    def generate(
        self,
        input_ids,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=None,
    ):
        import torch

        class DummyScores:
            def __init__(self):
                self.scores = [torch.zeros((1, 32128))]

        return DummyScores()

    def from_pretrained(*args, **kwargs):
        return DummyModel()


@pytest.fixture
def dummy_runner(monkeypatch):
    # Patch tokenizer and model
    monkeypatch.setattr("stance_detector.flant5_runner.T5Tokenizer", DummyTokenizer)
    monkeypatch.setattr(
        "stance_detector.flant5_runner.T5ForConditionalGeneration", DummyModel
    )
    runner = FlanT5Runner(
        model_name="dummy",
        tokenizer_cache="",
        model_cache="",
        device_map=None,
        max_memory={},
    )
    return runner


def test_run_inference_and_save(tmp_path, dummy_runner):
    # Create dummy data
    df = pd.DataFrame([{"Prompt": "test", "class_tokens": ["a", "b"]}])
    input_path = tmp_path / "input.parquet"
    df.to_parquet(input_path)
    # Test load_data
    loaded_df = dummy_runner.load_data(str(input_path))
    assert not loaded_df.empty

    # Test run_inference with interim and output
    output_path = tmp_path / "output.parquet"
    result_df = dummy_runner.run_inference(
        loaded_df,
        save_every=1,
        input_path=str(input_path),
        output_path=str(output_path),
    )
    assert not result_df.empty
    assert os.path.exists(output_path)

    # Test interim file
    interim_path = tmp_path / "input--interim_0_of_1.parquet"
    assert os.path.exists(interim_path)


def test_save_results(tmp_path, dummy_runner):
    df = pd.DataFrame([{"Prompt": "test", "class_tokens": ["a", "b"]}])
    input_path = tmp_path / "input.parquet"
    df.to_parquet(input_path)
    dummy_runner.save_results(df, str(input_path))
    output_path = tmp_path / "input--output.parquet"
    assert os.path.exists(output_path)
