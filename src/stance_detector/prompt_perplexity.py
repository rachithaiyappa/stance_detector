# @Author: Rachith Aiyappa
# @Date: 2025-05-04

from stance_detector.utils.lmppl import EncoderDecoderLM
import pandas as pd
import numpy as np
import pickle as pkl
from pathlib import Path
from typing import Union, Optional, List
from stance_detector.utils.logger import CustomLogger
from os.path import splitext, basename, dirname, join


class Perplexity:
    def __init__(
        self,
        model_name: str,
        cuda: int,
        hf_cache_dir: str = None,
        max_memory: int = None,
    ):
        self.logger = CustomLogger(__name__).get_logger()
        self.model = EncoderDecoderLM(
            model_name, hf_cache_dir=hf_cache_dir, cuda=cuda, max_memory=max_memory
        )
        self.logger.info(f"Model {model_name} loaded successfully.")

    def get_perplexity(
        self,
        input_file: Union[str, Path],
        prompt_col: str = "Prompt",
        output_path: Optional[Union[str, Path]] = None,
        batch_splits: int = 10,
        save_every: Optional[int] = None,
    ):
        # Load DataFrame if input_file is a path
        if isinstance(input_file, (str, Path)):
            df = pd.read_parquet(input_file)
        elif isinstance(input_file, pd.DataFrame):
            df = input_file
            self.logger.info(f"DataFrame with {len(df)} records loaded.")
        else:
            self.logger.error("input_file must be a path or a pandas DataFrame")
            raise ValueError("input_file must be a path or a pandas DataFrame")

        prompts = df[prompt_col].tolist()
        ppl: List = [None] * len(prompts)
        total = len(prompts)
        prompts_split = np.array_split(prompts, batch_splits)
        idx_split = np.array_split(np.arange(len(prompts)), batch_splits)
        for counter, (batch, idxs) in enumerate(zip(prompts_split, idx_split)):
            batch = list(batch)
            inputs = [""] * len(batch)
            self.logger.info(
                f"Processing batch {counter + 1}/{batch_splits} with {len(batch)} prompts."
            )
            # Get perplexity for the batch
            batch_ppl = self.model.get_perplexity(
                input_texts=inputs, output_texts=batch
            )
            for i, idx in enumerate(idxs):
                ppl[idx] = batch_ppl[i]
            if (
                save_every
                and save_every > 0
                and counter % save_every == 0
                and input_file
            ):
                base = splitext(basename(input_file))[0]
                dir_ = dirname(input_file)
                interim_path = join(dir_, f"{base}--interim_{counter}_of_{total}.pkl")
                # Save interim DataFrame with current ppl values
                interim_df = df.copy()
                interim_df["perplexity"] = ppl
                with open(interim_path, "wb") as f:
                    pkl.dump(interim_df, f)
                self.logger.info(f"Interim results saved to {interim_path}")

        df["perplexity"] = ppl
        if output_path:
            with open(output_path, "wb") as f:
                pkl.dump(df, f)
            self.logger.info(f"Final results saved to {output_path}")
        else:
            self.save_results(
                df,
                input_file,
            )
        self.logger.info(f"Processed {total} records")
        return ppl

    def save_results(self, df: pd.DataFrame, input_path: str) -> None:
        base = splitext(basename(input_path))[0]
        dir_ = dirname(input_path)
        output_path = join(dir_, f"{base}--perplexity.pkl")
        with open(output_path, "wb") as f:
            pkl.dump(df, f)
        self.logger.info(f"Saved results to {output_path}")


# Example usage:
if __name__ == "__main__":
    # Example arguments
    model_name = "your-model"
    cuda = 0
    parquet_path = (
        "../data/semeval_taskA_targetAtheism_prompt1_instruction1_prompts.parquet"
    )
    output_path = "./perplexities.pkl"

    scorer = Perplexity(model_name=model_name, cuda=cuda)
    perplexities = scorer.get_perplexity(
        input_file=parquet_path,
        prompt_col="Prompt",
        output_path=output_path,
        batch_splits=10,
    )
    print("Perplexities computed:", len(perplexities))
