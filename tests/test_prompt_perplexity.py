# @Author: Rachith Aiyappa
# @Date: 2025-05-04

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import pickle as pkl

from stance_detector.prompt_perplexity import Perplexity

class DummyModel:
    def get_perplexity(self, input_texts, output_texts):
        # Return a fixed perplexity for each prompt
        return [42.0 for _ in output_texts]

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Prompt": ["This is a test prompt.", "Another prompt for testing."]
    })

@pytest.fixture
def tmp_parquet(sample_df, tmp_path):
    parquet_path = tmp_path / "test_prompts.parquet"
    sample_df.to_parquet(parquet_path)
    return parquet_path

@pytest.fixture
def perplexity_instance(monkeypatch):
    # Patch EncoderDecoderLM to return DummyModel before Perplexity is instantiated
    from stance_detector import prompt_perplexity
    monkeypatch.setattr(prompt_perplexity, "EncoderDecoderLM", lambda *args, **kwargs: DummyModel())
    p = Perplexity(model_name="dummy", cuda=0)
    return p

def test_get_perplexity_with_path(perplexity_instance, tmp_parquet):
    ppl = perplexity_instance.get_perplexity(
        input_file=tmp_parquet,
        prompt_col="Prompt",
        batch_splits=2
    )
    assert isinstance(ppl, list)
    assert all(x == 42.0 for x in ppl)

def test_interim_and_final_save(perplexity_instance, tmp_parquet, tmp_path):
    output_path = tmp_path / "final.pkl"
    ppl = perplexity_instance.get_perplexity(
        input_file=tmp_parquet,
        prompt_col="Prompt",
        output_path=output_path,
        batch_splits=2,
        save_every=1
    )
    # Check final output file exists and is readable
    assert output_path.exists()
    with open(output_path, "rb") as f:
        df = pkl.load(f)
    assert "perplexity" in df.columns
    # Check at least one interim file exists
    interim_files = list(tmp_path.glob("*--interim_*.pkl"))
    assert len(interim_files) > 0