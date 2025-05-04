# @Author: Rachith Aiyappa
# @Date: 2025-05-03

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import re
from stance_detector.utils.logger import CustomLogger
from stance_detector.utils.argparse_utils import decoder_args
from os.path import splitext, basename, dirname, join
import pickle as pkl


class AFTDecoder:
    def __init__(self):
        self.logger = CustomLogger(__name__).get_logger()

    @staticmethod
    def filter_nones(list_of_dictionaries: List[Dict[str, Any]]) -> None:
        """Remove keys with NaN values from dictionaries in a list."""
        if not list_of_dictionaries:
            return
        for dict_ in list_of_dictionaries:
            keys_to_remove = [
                key for key, value in dict_.items() if value is None or np.isnan(value)
            ]
            for key in keys_to_remove:
                del dict_[key]

    @staticmethod
    def readd(list_of_dictionaries: List[Dict[str, float]]) -> List[float]:
        """Sum values in each dictionary in a list."""
        return [sum(i.values()) for i in list_of_dictionaries]

    @staticmethod
    def log_to_prob(x: float) -> float:
        """Convert log probabilities to probabilities."""
        if x == 0:
            return 0
        return np.exp(x)

    def save_results(self, df: pd.DataFrame, input_path: str) -> None:
        """Save DataFrame to a pickle file with a modified filename."""
        base = splitext(basename(input_path))[0]
        dir_ = dirname(input_path)
        base = base.replace("--output", "--aft")
        output_path = join(dir_, f"{base}-output.pkl")
        with open(output_path, "wb") as f:
            pkl.dump(df, f)
        self.logger.info(f"Saved results to {output_path}")

    def _extract_prompt_and_instruction(
        self,
        data: Union[Path, str],
        prompt_template: Optional[str],
        instruction: Optional[str],
    ) -> Tuple[str, str]:
        """Extract prompt_template and instruction from filename if not provided."""
        if prompt_template is None or instruction is None:
            self.logger.info(
                "One of prompt_template or instruction is not specified. "
                "Assuming filename to be of the kind *prompt3f_instruction1* "
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
        return prompt_template, instruction

    def _prepare_dataframe(
        self, data: Union[Path, str], context_free_data: Union[Path, str]
    ) -> pd.DataFrame:
        """Load and concatenate context and context-free DataFrames."""
        self.logger.info(f"Loading context data from: {data}")
        self.logger.info(f"Loading context-free data from: {context_free_data}")
        with open(data, "rb") as f:
            df_with_context = pkl.load(f)
        with open(context_free_data, "rb") as f:
            df_context_free = pkl.load(f)
        df = pd.concat([df_with_context, df_context_free], ignore_index=True)
        self.logger.info(f"Loaded {len(df)} rows after concatenation.")
        return df

    def _clean_and_sum_log_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean class_token_log_probs and compute their sums."""
        self.logger.info("Cleaning class_token_log_probs and computing totals...")
        df["class_token_log_probs"].apply(self.filter_nones)
        df["class_token_log_probs_total"] = df["class_token_log_probs"].apply(
            self.readd
        )
        return df

    def _compute_aft(self, df: pd.DataFrame, label1: str, label2: str) -> pd.DataFrame:
        """Compute AFT probabilities and labels."""
        self.logger.info("Computing AFT probabilities and labels...")
        df["aft_label"] = ""
        df["aft_class_token_probs"] = ""

        # Get normalization weights from context-free
        context_free_row = df[df.label == label2]
        if context_free_row.empty:
            self.logger.error(f"No context-free rows found for label: {label2}")
            raise ValueError(f"No context-free rows found for label: {label2}")
        class_token_log_probs_total_cf = np.array(
            context_free_row.class_token_log_probs_total.iloc[0]
        )
        class_token_probs_total_cf = np.array(
            [self.log_to_prob(x) for x in class_token_log_probs_total_cf]
        )
        w = np.diag(1 / class_token_probs_total_cf)

        # Compute for each context row
        for i in df[df.label == label1].index:
            context_row = df.loc[i]
            class_token_log_probs_total = np.array(
                [self.log_to_prob(x) for x in context_row.class_token_log_probs_total]
            )
            aft_logits = np.dot(w, class_token_log_probs_total)
            softmax = np.exp(aft_logits) / np.sum(np.exp(aft_logits))
            df.at[i, "aft_class_token_probs"] = softmax
            aft_class_token_index = np.argmax(softmax)
            df.at[i, "aft_label"] = context_row.class_tokens[aft_class_token_index]
            self.logger.debug(
                f"Row {i}: AFT label = {df.at[i, 'aft_label']}, probs = {softmax}"
            )
        return df

    def decode(
        self,
        data: Union[Path, str],
        context_free_data: Union[Path, str],
        output_path: Optional[Union[Path, str]] = None,
        prompt_template: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run the AFT decoding pipeline and return the processed DataFrame."""
        df = self._prepare_dataframe(data, context_free_data)
        df = self._clean_and_sum_log_probs(df)
        prompt_template, instruction = self._extract_prompt_and_instruction(
            data, prompt_template, instruction
        )
        self.logger.info(
            f"Filtering data for prompt_template: {prompt_template} and instruction: {instruction}"
        )
        label1 = f"{prompt_template}--{instruction}"
        label2 = f"{prompt_template}_free--{instruction}"
        df = df[(df.label == label1) | (df.label == label2)].copy()
        df = self._compute_aft(df, label1, label2)

        if output_path:
            with open(output_path, "wb") as f:
                pkl.dump(df, f)
            self.logger.info(f"Final results saved to {output_path}")
        else:
            self.save_results(df, data)
        self.logger.info(f"Processed {len(df)} records")
        return df


def main():
    args = decoder_args.parse_args()
    decoder = AFTDecoder()
    decoder.decode(args.data, args.context_free_data, args.output)


if __name__ == "__main__":
    main()
