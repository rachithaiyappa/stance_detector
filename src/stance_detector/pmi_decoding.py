# @Author: Rachith Aiyappa
# @Date: 2025-05-01

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import re
from stance_detector.utils.logger import CustomLogger
from stance_detector.utils.argparse_utils import decoder_args
from os.path import splitext, basename, dirname, join
import pickle as pkl


class PMIDecoder:
    def __init__(self):
        self.logger = CustomLogger(__name__).get_logger()

    @staticmethod
    def filter_nones(list_of_dictionaries: List[Dict[str, Any]]) -> None:
        """Remove keys with NaN values from dictionaries in a list."""
        if list_of_dictionaries is None or len(list_of_dictionaries) == 0:
            return
        for dict_ in list_of_dictionaries:
            keys_to_remove = [
                key for key, value in dict_.items() if value is None or np.isnan(value)
            ]
            for key in keys_to_remove:
                del dict_[key]

    @staticmethod
    def readd(list_of_dictionaries: List[Dict[str, float]]) -> float:
        """Sum values in each dictionary in a list."""
        return [sum(i.values()) for i in list_of_dictionaries]

    @staticmethod
    def explode_and_merge(df: pd.DataFrame, list_cols: List[str]) -> pd.DataFrame:
        """Explode list columns and merge with other columns."""
        exploded = {col: df[col].explode() for col in list_cols}
        other_cols = list(set(df.columns) - set(list_cols))
        df2 = pd.DataFrame(exploded)
        df2 = df[other_cols].merge(df2, how="right", left_index=True, right_index=True)
        df2 = df2.loc[:, df.columns]
        return df2

    @staticmethod
    def deduplicate_context_pmi(context_pmi: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate by ID_x or ID, keeping one sample per group."""
        if "ID_x" in context_pmi.columns:
            return context_pmi.groupby("ID_x", as_index=False).head(1)
        elif "ID" in context_pmi.columns:
            return context_pmi.groupby("ID", as_index=False).head(1)
        return context_pmi

    def save_results(self, df: pd.DataFrame, input_path: str) -> None:
        base = splitext(basename(input_path))[0]
        dir_ = dirname(input_path)
        base = base.replace("--output", "--pmi")
        output_path = join(dir_, f"{base}-output.pkl")
        with open(output_path, "wb") as f:
            pkl.dump(df, f)
        self.logger.info(f"Saved results to {output_path}")

    def decode(
        self,
        data: Union[Path, str],
        context_free_data: Union[Path, str],
        output_path: Optional[Union[Path, str]] = None,
        prompt_template: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run the PMI decoding pipeline and return the processed DataFrame."""

        self.logger.info(f"Loading context data from: {data}")
        self.logger.info(f"Loading context-free data from: {context_free_data}")

        with open(data, "rb") as f:
            df_with_context = pkl.load(f)
        with open(context_free_data, "rb") as f:
            df_context_free = pkl.load(f)
        df = pd.concat([df_with_context, df_context_free], ignore_index=True)

        self.logger.info("Cleaning class_token_log_probs...")
        df["class_token_log_probs"].apply(self.filter_nones)
        df["class_token_log_probs_total"] = df["class_token_log_probs"].apply(
            self.readd
        )

        if prompt_template is None or instruction is None:
            self.logger.info(
                "One of prompt_template or instruction is not specified. "
                "Assuming filename to be of the kind *prompt3f_instruction1*"
                "and extracting prompt_template (3f) and instruction (1) from filename..."
            )
            match = re.search(r"prompt([a-zA-Z0-9]+)_instruction(\d+)", Path(data).stem)
            if not match:
                self.logger.error(
                    f"Could not extract prompt/instruction from filename: {data}"
                )
                raise ValueError(
                    f"Could not extract prompt/instruction from filename: {data}"
                )
            prompt_template = match.group(1)
            instruction = match.group(2)
        else:
            self.logger.info(
                f"Using prompt_template: {prompt_template} and instruction: {instruction}"
            )
        self.logger.info(
            f"Filtering data for prompt_template: {prompt_template} and instruction: {instruction}"
        )
        label1 = f"{prompt_template}--{instruction}"
        label2 = f"{prompt_template}_free--{instruction}"
        df = df[(df.label == label1) | (df.label == label2)]

        list_cols = ["class_tokens", "class_token_log_probs_total"]
        df2 = self.explode_and_merge(df, list_cols)

        df2_free = df2[df2["label"].str.contains(f"{prompt_template}_free")].copy()
        df2_free_label = df2_free[df2_free.label == label2].copy()
        context_filt = df2.merge(
            df2_free_label,
            right_on=["Target", "class_tokens"],
            left_on=["Target", "class_tokens"],
            suffixes=("", "_free"),
        )
        context_filt = context_filt[context_filt["label"] == label1].reset_index(
            drop=True
        )

        # Select relevant columns
        keep_cols = [
            "ID",
            "Target",
            "Tweet",
            "label",
            "Stance",
            "Prompt",
            "Prompt_free",
            "class_tokens",
            "class_token_log_probs_total",
            "class_token_log_probs_total_free",
            "generated_token",
            "generated_token_free",
            "generated_overall_log_prob",
            "generated_overall_log_prob_free",
        ]
        context_filt = context_filt[
            [col for col in keep_cols if col in context_filt.columns]
        ]

        # core PMI logic
        context_filt["pmi_lognum-logden"] = (
            context_filt["class_token_log_probs_total"]
            - context_filt["class_token_log_probs_total_free"]
        )

        idx = (
            context_filt.groupby(["ID"])["pmi_lognum-logden"].transform(max)
            == context_filt["pmi_lognum-logden"]
        )
        context_pmi = context_filt[idx].copy().sort_values(by=["ID"])
        context_pmi = self.deduplicate_context_pmi(context_pmi)

        if output_path:
            with open(output_path, "wb") as f:
                pkl.dump(pd.DataFrame(context_pmi), f)
            self.logger.info(f"Final results saved to {output_path}")
        else:
            self.save_results(pd.DataFrame(context_pmi), data)
        self.logger.info(f"Processed {len(context_pmi)} records")

        return context_pmi


def main():
    args = decoder_args.parse_args()
    decoder = PMIDecoder()
    decoder.decode(args.data, args.context_free_data, args.output)


if __name__ == "__main__":
    main()
