# @Author: Rachith Aiyappa
# @Date: 2025-04-30

import pandas as pd
from typing import List, Dict, Optional, Union
from pathlib import Path
from stance_detector.prompt_config import (
    INSTRUCTION_TEMPLATES,
    PROMPT_TEMPLATES,
    INSTRUCTION_OPTIONS,
    FLANT5_TOKENS,
)
from stance_detector.utils.logger import CustomLogger
from stance_detector.utils.argparse_utils import prompt_builder_args


class PromptBuilder:
    def __init__(self, csv_path: str):
        self.logger = CustomLogger(__name__).get_logger()
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.is_semeval = "semeval" in csv_path.lower()
        self.is_pstance = "pstance" in csv_path.lower()
        self.logger.info(f"Loaded CSV file: {csv_path}")

    def get_ops(self, template_key: str) -> List[str]:
        """Get the options for the instruction referenced by the template_key."""
        ops = INSTRUCTION_OPTIONS[template_key]
        if self.is_pstance:
            # Remove all options containing 'neutral' for pstance datasets
            return [op for op in ops if "neutral" not in op.lower()]
        return ops

    def get_class_tokens(self, template_key: str) -> List[str]:
        """Class tokens for which we extract probabilities for downstream analysis."""
        tokens = FLANT5_TOKENS[template_key]
        if self.is_pstance:
            # Remove all tokens containing 'neutral' for pstance datasets
            return [tok for tok in tokens if "neutral" not in tok.lower()]
        return tokens

    def format_instruction(self, instr_template: str, ops: List[str]) -> str:
        """
        Format the instruction template with the provided options.
        """
        # If only two options, remove {op3} and the preceding comma/or
        if len(ops) == 2:
            instr = instr_template
            instr = instr.replace(", or {op3}.", ".")
            instr = instr.replace(" or {op3}.", ".")
            instr = instr.replace(", {op2}", ", or {op2}.")
            return instr.format(op1=ops[0], op2=ops[1], op3="")
        else:
            return instr_template.format(op1=ops[0], op2=ops[1], op3=ops[2])

    def build_all_prompts(
        self,
        targets: Optional[Union[List[str], str]] = None,
        output_path: Optional[str] = None,
        prompt_template_key: Optional[str] = None,
        instruction_key: Optional[str] = None,
        context_free: bool = False,
    ) -> pd.DataFrame:
        """
        Build prompts for the specified prompt_template_key and instruction_key.
        If either is None, build for all available options.
        If context_free is True, the tweet will be omitted from the prompt.
        """
        df_filtered = self._filter_targets(targets)
        prompt_templates = self._select_templates(
            prompt_template_key, PROMPT_TEMPLATES, "prompt"
        )
        instruction_templates = self._select_templates(
            instruction_key, INSTRUCTION_TEMPLATES, "instruction"
        )

        all_rows = []
        for template_key, template in prompt_templates.items():
            ops = self.get_ops(template_key)
            class_tokens = self.get_class_tokens(template_key)
            for instr_key, instr_template in instruction_templates.items():
                instruction = self.format_instruction(instr_template, ops)
                df_prompts = self._build_prompts_for_template(
                    df_filtered,
                    template,
                    template_key,
                    instruction,
                    instr_key,
                    class_tokens,
                    context_free,
                )
                all_rows.append(df_prompts)

        result_df = pd.concat(all_rows, ignore_index=True)
        self._log_prompt_building(context_free)
        self._save_prompt_combinations(
            result_df=result_df,
            df_filtered=df_filtered,
            targets=targets,
            output_path=output_path,
            prompt_templates=prompt_templates,
            instruction_templates=instruction_templates,
            context_free=context_free,
        )
        return result_df

    def _filter_targets(self, targets: Optional[Union[List[str], str]]) -> pd.DataFrame:
        """Filter dataframe for specified targets if provided."""
        if targets is not None:
            if isinstance(targets, str):
                targets = [targets]
            df_filtered = self.df[self.df["Target"].isin(targets)].copy()
            self.logger.info(f"Filtered DataFrame for all targets: {targets}")
        else:
            self.logger.warning("No targets provided, using all available targets.")
            self.logger.info("Filtered DataFrame for all targets")
            df_filtered = self.df.copy()
        return df_filtered

    def _select_templates(self, key: Optional[str], templates: Dict, name: str) -> Dict:
        """Select templates based on the provided key."""
        if not key:
            self.logger.warning(
                f"No {name} template key provided, using all available {name} templates."
            )
        selected = {key: templates[key]} if key is not None else templates
        self.logger.info(f"Selected {name} templates: {list(selected.keys())}")
        return selected

    def _build_prompts_for_template(
        self,
        df_filtered: pd.DataFrame,
        template: str,
        template_key: str,
        instruction: str,
        instr_key: str,
        class_tokens: List[str],
        context_free: bool,
    ) -> pd.DataFrame:
        """Build prompts and labels for a given template/instruction combination."""
        if context_free:
            row = df_filtered.iloc[0]
            prompt = template.format(
                instruction=instruction,
                tweet="",
                target=row["Target"],
            )
            prompts = [prompt]
            labels = [f"{template_key}_free--{instr_key}"]
            df_copy = df_filtered.iloc[[0]].copy()
        else:
            prompts = []
            labels = []
            for _, row in df_filtered.iterrows():
                prompt = template.format(
                    instruction=instruction,
                    tweet=row["Tweet"],
                    target=row["Target"],
                )
                prompts.append(prompt)
                labels.append(f"{template_key}--{instr_key}")
            df_copy = df_filtered.copy()
        df_copy["Prompt"] = prompts
        df_copy["label"] = labels
        df_copy["class_tokens"] = [class_tokens] * len(df_copy)
        return df_copy

    def _log_prompt_building(self, context_free: bool):
        if context_free:
            self.logger.info(
                "Context-free prompts built successfully. Tweets omitted from prompts."
            )
        else:
            self.logger.info("Prompts built successfully with tweets included.")

    def _save_prompt_combinations(
        self,
        result_df: pd.DataFrame,
        df_filtered: pd.DataFrame,
        targets: Optional[Union[List[str], str]],
        output_path: Optional[str],
        prompt_templates: Dict,
        instruction_templates: Dict,
        context_free: bool = False,
    ):
        """Save prompt combinations to disk."""
        targets_list = (
            targets if targets is not None else sorted(df_filtered["Target"].unique())
        )
        if isinstance(targets_list, str):
            targets_list = [targets_list]

        prompt_keys = list(prompt_templates.keys())
        instr_keys = list(instruction_templates.keys())

        for t in targets_list:
            for p in prompt_keys:
                for i in instr_keys:
                    df_combo = result_df.copy()
                    if t is not None:
                        df_combo = df_combo[df_combo["Target"] == t]
                    if p is not None:
                        df_combo = df_combo[
                            df_combo["label"].str.startswith(f"{p}--")
                            | df_combo["label"].str.startswith(f"{p}_free--")
                        ]
                    if i is not None:
                        df_combo = df_combo[df_combo["label"].str.endswith(f"--{i}")]
                    if df_combo.empty:
                        continue

                    dataset_name = self.csv_path.split("/")[-1].split(".")[0]
                    parts = [dataset_name]
                    if t is not None:
                        parts.append(f'target{str(t).replace(" ", ":")}')
                    if p is not None:
                        parts.append(f"prompt{str(p)}")
                    if i is not None:
                        parts.append(f"instruction{str(i)}")
                    filename = "_".join(parts) + "_prompts"
                    if context_free:
                        filename += "_free"
                    filename += ".parquet"

                    if output_path is None:
                        root_dir = Path(__file__).resolve().parents[2]
                        out_path = root_dir / "data" / filename
                    else:
                        out_path = Path(output_path).parent / filename

                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Saving prompts to: {out_path}")
                    df_combo.to_parquet(out_path)


def main():
    args = prompt_builder_args.parse_args()
    builder = PromptBuilder(args.csv_path)
    return builder.build_all_prompts(
        targets=args.targets,
        output_path=args.output_path,
        prompt_template_key=args.prompt_template_key,
        instruction_key=args.instruction_key,
        context_free=args.context_free,
    )


if __name__ == "__main__":
    main()
