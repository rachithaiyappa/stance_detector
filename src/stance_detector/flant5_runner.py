# @Author: Rachith Aiyappa
# @Date: 2025-05-01

from stance_detector.utils.logger import CustomLogger
import pandas as pd
import torch
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path
from typing import Optional, List, Any, Dict, Tuple
from os.path import splitext, basename, dirname, join
from stance_detector.utils.argparse_utils import (
    get_flan_t5_inference_args,
    get_flan_t5_model_args,
)
import yaml
from tqdm import tqdm
import pickle as pkl


class FlanT5Runner:
    def __init__(
        self,
        model_name: str,
        tokenizer_cache: str,
        model_cache: str,
        device_map: Any,
        max_memory: dict,
    ) -> None:
        self.logger = CustomLogger(__name__).get_logger()
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name, cache_dir=tokenizer_cache
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=model_cache,
            device_map=device_map,
            max_memory=max_memory,
        )

    def load_data(self, input_path: str) -> pd.DataFrame:
        ext = Path(input_path).suffix
        if ext == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        self.logger.info(f"Loaded {len(df)} records from {input_path}")
        return df

    def run_inference(
        self,
        save_every: Optional[int] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run inference on the provided DataFrame.
        """
        df = self.load_data(input_path)
        results: List[Dict[str, Any]] = []
        interim_results: List[Dict[str, Any]] = []
        data = df.to_dict(orient="records")
        total = len(data)
        for counter, item in tqdm(enumerate(data), total=len(data)):
            text = item["Prompt"]
            class_tokens = item["class_tokens"]
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
            scores = self.model.generate(
                input_ids, return_dict_in_generate=True, output_scores=True
            ).scores

            # greedy decoding. selects the token
            # (among all tokens of the vocabulary with the logit score)
            item.update(
                {
                    "generated_token": self.tokenizer.decode(
                        [torch.argmax(i) for i in scores], skip_special_tokens=True
                    )
                }
            )
            item.update(
                {
                    "generated_out_tokens_log_prob": self._get_generated_out_tokens_log_prob(
                        scores
                    )
                }
            )
            item.update(
                {
                    "generated_overall_log_prob": float(
                        sum(item["generated_out_tokens_log_prob"].values())
                    )
                }
            )

            (
                class_token_log_probs,
                class_token_log_probs_total,
            ) = self._get_class_token_log_probs(text, class_tokens)
            item.update({"class_token_log_probs": class_token_log_probs})
            item.update({"class_token_log_probs_total": class_token_log_probs_total})
            results.append(item)
            interim_results.append(item)
            if (
                save_every
                and save_every > 0
                and counter % save_every == 0
                and input_path
            ):
                base = splitext(basename(input_path))[0]
                dir_ = dirname(input_path)
                interim_path = join(dir_, f"{base}--interim_{counter}_of_{total}.pkl")
                with open(interim_path, "wb") as f:
                    pkl.dump(pd.DataFrame(interim_results), f)
                self.logger.info(f"Interim results saved to {interim_path}")
                interim_results = []
            if (counter + 1) % 100 == 0:
                self.logger.info(f"Processed {counter+1} records")
        if output_path:
            with open(output_path, "wb") as f:
                pkl.dump(pd.DataFrame(results), f)
            self.logger.info(f"Final results saved to {output_path}")
        else:
            self.save_results(
                pd.DataFrame(results),
                input_path,
            )
        self.logger.info(f"Processed {total} records")
        return pd.DataFrame(results)

    def _get_generated_out_tokens_log_prob(self, scores: Any) -> Dict[str, float]:
        """
        Returns a dictionary of generated tokens and their log probabilities.
        """
        out = {}
        for i in scores:
            token_id = torch.argmax(i)
            token = self.tokenizer.decode(token_id)
            log_prob = float(
                torch.log(torch.max(torch.nn.functional.softmax(i, dim=1)))
            )
            out[token] = log_prob
        return out

    def _get_class_token_log_probs(
        self, text: str, class_tokens: List[str]
    ) -> Tuple[List[Dict[str, float]], List[float]]:
        """
        Returns a list of dictionaries, each containing the log probabilities of
        words/class tokens for a given text.
        The first element of the tuple is a list
        of dictionaries, where each dictionary corresponds to a word and
        contains the log probabilities of that tokens forming that token.
        The second element is a list of log probabilities for each word.
        This is obtaiend from summing up the word's token probabilities.
        """

        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        class_token_log_probs = []
        class_token_log_probs_total = []
        ct_ids = [
            self.tokenizer(ct, return_tensors="pt").input_ids for ct in class_tokens
        ]
        for ct_id in ct_ids:
            temp_dict = {}
            scores = self.model.generate(
                input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=len(ct_id[0]),
            ).scores
            for pos, ct in enumerate(ct_id[0]):
                try:
                    prob = torch.nn.functional.softmax(scores[pos][0], dim=0)[ct]
                    temp_dict[self.tokenizer.decode(ct)] = float(torch.log(prob))
                except IndexError:
                    temp_dict[self.tokenizer.decode(ct)] = np.nan
            class_token_log_probs_total.append(float(sum(temp_dict.values())))
            class_token_log_probs.append(temp_dict)
        return class_token_log_probs, class_token_log_probs_total

    # PERHAPS MORE EFFICIENT
    # def _get_class_token_log_probs(
    #     self, text: str, class_tokens: List[str]
    # ) -> Tuple[List[Dict[str, float]], List[float]]:
    #     input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

    #     class_token_log_probs = []
    #     class_token_log_probs_total = []

    #     for ct in class_tokens:
    #         ct_inputs = self.tokenizer(ct, return_tensors="pt").input_ids.to(self.model.device)

    #         # Concatenate input with class token
    #         full_input = torch.cat([input_ids, ct_inputs], dim=1)
    #         with torch.no_grad():
    #             outputs = self.model(input_ids=full_input)
    #             logits = outputs.logits[:, -ct_inputs.shape[1]-1:-1, :]  # last few logits before ct tokens

    #         temp_dict = {}
    #         for i, ct_token_id in enumerate(ct_inputs[0]):
    #             token_logits = logits[0, i]
    #             prob = torch.nn.functional.softmax(token_logits, dim=0)[ct_token_id]
    #             decoded_token = self.tokenizer.decode([ct_token_id])
    #             temp_dict[decoded_token] = float(torch.log(prob))

    #         class_token_log_probs_total.append(float(sum(temp_dict.values())))
    #         class_token_log_probs.append(temp_dict)

    #     return class_token_log_probs, class_token_log_probs_total

    def save_results(self, df: pd.DataFrame, input_path: str) -> None:
        base = splitext(basename(input_path))[0]
        dir_ = dirname(input_path)
        output_path = join(dir_, f"{base}--output.pkl")
        with open(output_path, "wb") as f:
            pkl.dump(df, f)
        self.logger.info(f"Saved results to {output_path}")


def main():
    model_parser = get_flan_t5_model_args()
    model_args = model_parser.parse_args()
    inference_parser = get_flan_t5_inference_args()
    inference_args = inference_parser.parse_args()

    # Load config.yaml
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    max_memory_mapping = config["max_memory_mapping"]

    runner = FlanT5Runner(
        model_name=model_args.model_name,
        tokenizer_cache=model_args.tokenizer_cache,
        model_cache=model_args.model_cache,
        device_map=model_args.device_map,
        max_memory=max_memory_mapping,
    )

    runner.run_inference(
        save_every=inference_args.save_every if inference_args.save_every > 0 else None,
        input_path=inference_args.input,
        output_path=inference_args.output,
    )


if __name__ == "__main__":
    main()
