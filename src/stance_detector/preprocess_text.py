# @Author: Rachith Aiyappa
# @Date: 2025-04-30

import pandas as pd
from typing import Optional, Union
from pathlib import Path
from stance_detector.utils.logger import CustomLogger
from stance_detector.utils.preprocessor import normalizeTweet
from stance_detector.utils.argparse_utils import preprocess_args


class PreProcessText:
    def __init__(self):
        self.logger = CustomLogger(__name__).get_logger()
        self.logger.info("Initiated PreProcessor")

    def preprocess_text(
        self,
        input_path: Union[Path, str] = None,
        output_path: Optional[Union[Path, str]] = None,
    ) -> pd.DataFrame:

        self.csv_path = input_path
        self.df = pd.read_csv(input_path)
        # self.is_semeval = "semeval" in input_path.lower()
        # self.is_pstance = "pstance" in input_path.lower()
        self.logger.info(f"Loaded CSV file: {input_path}")

        self.df["Tweet"] = self.df["Tweet"].apply(normalizeTweet)

        self._save_preprocessed_tweet(
            output_path=output_path,
        )
        return self.df

    def _save_preprocessed_tweet(
        self, output_path: Optional[Union[Path, str]] = None
    ) -> None:
        if output_path is None:
            root_dir = Path(__file__).resolve().parents[2]
            filename = self.csv_path.stem + "_preprocessed.csv"
            output_path = root_dir / "data" / filename

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving preprocessed tweets to: {output_path}")
        self.df.to_csv(output_path, index=False)


def main():
    args = preprocess_args.parse_args()
    preprocess = PreProcessText()
    return preprocess.preprocess_text(
        input_path=args.input_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
