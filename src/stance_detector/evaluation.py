# @Author: Rachith Aiyappa
# @Date: 2025-05-02

import pandas as pd
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
from os.path import splitext, basename, dirname, join
from stance_detector.utils.logger import CustomLogger
from stance_detector.utils.argparse_utils import evaluation_args
import pickle as pkl

class Evaluator:
    def __init__(self):
        self.logger = CustomLogger(__name__).get_logger()
        self.evaluation_function = {
            "1": self.mapping_positive_negative,
            "2": self.mapping_favor_against,
            "3f": self.mapping_favor,
            "3a": self.mapping_against,
            "4f_not": self.mapping_favor,
            "4a_not": self.mapping_against,
        }

    @staticmethod
    def mapping_positive_negative(i):
        if "positive" in str(i).lower():
            return "FAVOR"
        elif "negative" in str(i).lower():
            return "AGAINST"
        else:
            return "NONE"

    @staticmethod
    def mapping_favor(i):
        if "true" in str(i).lower():
            return "FAVOR"
        elif "false" in str(i).lower():
            return "AGAINST"
        else:
            return "NONE"

    @staticmethod
    def mapping_against(i):
        if "true" in str(i).lower():
            return "AGAINST"
        elif "false" in str(i).lower():
            return "FAVOR"
        else:
            return "NONE"

    @staticmethod
    def mapping_favor_against(i):
        if "favor" in str(i).lower():
            return "FAVOR"
        elif "against" in str(i).lower():
            return "AGAINST"
        else:
            return "NONE"

    @staticmethod
    def evaluation_against(df):
        tp = len(df[(df["Stance"] == "AGAINST") & (df["Assigned"] == "AGAINST")])
        fp = len(df[(df["Stance"] != "AGAINST") & (df["Assigned"] == "AGAINST")])
        tn = len(df[(df["Stance"] != "AGAINST") & (df["Assigned"] != "AGAINST")])
        fn = len(df[(df["Stance"] == "AGAINST") & (df["Assigned"] != "AGAINST")])
        return tp, fp, tn, fn

    @staticmethod
    def evaluation_favor(df):
        tp = len(df[(df["Stance"] == "FAVOR") & (df["Assigned"] == "FAVOR")])
        fp = len(df[(df["Stance"] != "FAVOR") & (df["Assigned"] == "FAVOR")])
        tn = len(df[(df["Stance"] != "FAVOR") & (df["Assigned"] != "FAVOR")])
        fn = len(df[(df["Stance"] == "FAVOR") & (df["Assigned"] != "FAVOR")])
        return tp, fp, tn, fn

    def save_results(self, df: pd.DataFrame, input_path: str) -> None:
        base = splitext(basename(input_path))[0]
        dir_ = dirname(input_path)
        output_path = join(dir_, f"{base}-eval.csv")
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved results to {output_path}")

    def evaluate(
        self,
        input_path: Union[Path, str],
        prompt_template: Optional[str] = None,
        instruction: Optional[str] = None,
        output_path: Optional[Union[Path, str]] = None,
        decoding: str = "greedy",
    ) -> Dict[str, Any]:
        with open(input_path, "rb") as f:
            df = pkl.load(f)
        if decoding == "greedy":
            df.rename(columns={"generated_token": "Assigned"}, inplace=True)
            self.logger.info("Evaluating greedy decoding")
        elif decoding == "pmi":
            df.rename(columns={"class_tokens": "Assigned"}, inplace=True)
            self.logger.info("Evaluating pmi decoding")
        elif decoding == "aft":
            df.rename(columns={"aft_label": "Assigned"}, inplace=True)
            self.logger.info("Evaluating aft decoding")
        else:
            self.logger.error(f"Unknown decoding method: {decoding}")
            raise ValueError(f"Unknown decoding method: {decoding}")

        # Extract prompt_template and instruction if not provided
        if prompt_template is None or instruction is None:
            self.logger.info(
                "One of prompt_template or instruction is not specified. "
                "Assuming filename to be of the kind *prompt3f_instruction1*"
                " and extracting prompt_template (3f) and instruction (1) from filename..."
            )
            match = re.search(r"prompt([a-zA-Z0-9]+)_instruction(\d+)", Path(input_path).stem)
            if not match:
                self.logger.error(
                    f"Could not extract prompt/instruction from filename: {Path(input_path).stem}"
                )
                raise ValueError(
                    f"Could not extract prompt/instruction from filename: {Path(input_path).stem}"
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

        # Map generated tokens to stance labels
        mapping_func = self.evaluation_function.get(prompt_template)
        if mapping_func is None:
            raise ValueError(f"No evaluation function for prompt_template: {prompt_template}")
        df["Assigned"] = df["Assigned"].map(mapping_func)

        # Calculate metrics for AGAINST
        tp, fp, tn, fn = self.evaluation_against(df)
        against_f1 = 100 * 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        # Calculate metrics for FAVOR
        tp_f, fp_f, tn_f, fn_f = self.evaluation_favor(df)
        favor_f1 = 100 * 2 * tp_f / (2 * tp_f + fp_f + fn_f) if (2 * tp_f + fp_f + fn_f) > 0 else 0

        # Prepare results DataFrame
        results_df = pd.DataFrame(
            {
                "prompt_template": [prompt_template],
                "instruction": [instruction],
                "favor_tp": [tp_f],
                "favor_fp": [fp_f],
                "favor_tn": [tn_f],
                "favor_fn": [fn_f],
                "favor_f1": [favor_f1],
                "against_tp": [tp],
                "against_fp": [fp],
                "against_tn": [tn],
                "against_fn": [fn],
                "against_f1": [against_f1],
                "overall_f1": [(favor_f1 + against_f1)/2]
            }
        )

        # Save results
        if output_path:
            results_df.to_csv(output_path, index=False)
            self.logger.info(f"Final results saved to {output_path}")
        else:
            self.save_results(results_df, input_path)

        self.logger.info(f"Processed evaluation for {len(df)} records")
        return results_df.to_dict(orient="records")[0]

def main():
    args = evaluation_args.parse_args()
    evaluator = Evaluator()
    evaluator.evaluate(args.input, args.prompt_template, args.instruction, args.output)

if __name__ == "__main__":
    main()
