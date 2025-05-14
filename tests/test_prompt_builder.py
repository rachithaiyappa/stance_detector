# @Author: Rachith Aiyappa
# @Date: 2025-04-30

import shutil
import pandas as pd
import pytest
from pathlib import Path
from stance_detector.prompt_builder import PromptBuilder
from stance_detector.prompt_config import (
    PROMPT_TEMPLATES,
    INSTRUCTION_TEMPLATES,
    INSTRUCTION_OPTIONS,
)

TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    # Use the first prompt template's options for test data
    first_prompt_key = list(INSTRUCTION_OPTIONS.keys())[0]
    options = INSTRUCTION_OPTIONS[first_prompt_key]
    targets = ["Atheism", "Climate Change", "Atheism"]
    stances = [options[0], options[1], options[2] if len(options) > 2 else options[1]]
    tmp_dir = tmp_path_factory.mktemp("data")
    csv_path = tmp_dir / "sample_semeval.csv"
    df = pd.DataFrame(
        {
            "Target": targets,
            "Tweet": ["Tweet 1", "Tweet 2", "Tweet 3"],
            "Stance": stances,
        }
    )
    df.to_csv(csv_path)
    return csv_path


@pytest.fixture(scope="function", autouse=True)
def cleanup_output():
    # Clean up output directory before and after each test
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    yield
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)


def test_build_all_prompts_all_combinations(sample_csv):
    builder = PromptBuilder(str(sample_csv))
    df = builder.build_all_prompts(output_path=str(TEST_OUTPUT_DIR / "all.parquet"))
    # Check that the output directory contains parquet files for all combinations
    assert any(TEST_OUTPUT_DIR.glob("*.parquet"))
    assert not df.empty


def test_build_prompts_for_specific_target(sample_csv):
    builder = PromptBuilder(str(sample_csv))
    target = "Atheism"
    df = builder.build_all_prompts(
        targets=target, output_path=str(TEST_OUTPUT_DIR / "atheism.parquet")
    )
    # Should only contain prompts for "Atheism"
    assert all(df["Target"] == target)
    assert any(TEST_OUTPUT_DIR.glob(f"*{target.replace(' ', '_')}*.parquet"))


def test_build_prompts_for_specific_prompt_and_instruction(sample_csv):
    builder = PromptBuilder(str(sample_csv))
    prompt_key = list(PROMPT_TEMPLATES.keys())[0]
    instr_key = list(INSTRUCTION_TEMPLATES.keys())[0]
    df = builder.build_all_prompts(
        prompt_template_key=prompt_key,
        instruction_key=instr_key,
        output_path=str(TEST_OUTPUT_DIR / "specific.parquet"),
    )
    # Should only contain prompts for the specified prompt and instruction
    assert all(df["label"].str.startswith(f"{prompt_key}--"))
    assert all(df["label"].str.endswith(f"--{instr_key}"))
    assert any(
        TEST_OUTPUT_DIR.glob(f"*prompt{prompt_key}_instruction{instr_key}*.parquet")
    )


def test_build_prompts_for_multiple_targets(sample_csv):
    builder = PromptBuilder(str(sample_csv))
    targets = ["Atheism", "Climate Change"]
    df = builder.build_all_prompts(
        targets=targets, output_path=str(TEST_OUTPUT_DIR / "multi.parquet")
    )
    assert set(df["Target"].unique()) == set(targets)
    for t in targets:
        assert any(TEST_OUTPUT_DIR.glob(f"*{t.replace(' ', ':')}*.parquet"))


def test_build_prompts_context_free_flag(sample_csv):
    """Test that context_free flag produces prompts with empty tweet and correct filename."""
    builder = PromptBuilder(str(sample_csv))
    df = builder.build_all_prompts(
        output_path=str(TEST_OUTPUT_DIR / "context_free.parquet"),
        context_free=True,
    )
    # All prompts should have empty tweet in the prompt string
    assert all([row == "" or "Tweet" not in row for row in df["Prompt"]])
    # Output files should have _free before .parquet
    assert any(
        str(p).endswith("_free.parquet") for p in TEST_OUTPUT_DIR.glob("*.parquet")
    )
    # Label should have _free in it
    assert all("_free" in label for label in df["label"].unique())
