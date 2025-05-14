import argparse


def preprocess_args():
    parser = argparse.ArgumentParser(description="Prompt Builder")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the CSV file"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the output preprocessed text (optional)",
    )

    return parser


def prompt_builder_args():
    parser = argparse.ArgumentParser(description="Prompt Builder")
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to the CSV file"
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        help="List of target strings to filter (optional)",
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to save the output prompts (optional)"
    )
    parser.add_argument(
        "--prompt_template_key",
        type=str,
        help="Key for the prompt template to use (optional)",
    )
    parser.add_argument(
        "--instruction_key",
        type=str,
        help="Key for the instruction template to use (optional)",
    )

    parser.add_argument(
        "--context_free",
        action="store_true",
        help="Flag to indicate if context-free prompts should be generated",
    )
    return parser


def get_flan_t5_model_args():
    parser = argparse.ArgumentParser(description="Flan-T5 Model Arguments")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        default="google/flan-t5-xxl",
        help="Name of the Flan-T5 model to use",
    )
    parser.add_argument(
        "--tokenizer-cache",
        type=str,
        required=True,
        default="/home/racball/flan-t5-xxl--tokeniser",
        help="Path to cache directory for tokenizer",
    )
    parser.add_argument(
        "--model-cache",
        type=str,
        required=True,
        default="/home/racball/models--flan-t5-xxl",
        help="Path to cache directory for model",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        required=False,
        default="sequential",
        help="Device map for model loading (e.g., 'sequential', 'auto')",
    )
    return parser


def get_flan_t5_inference_args():
    parser = argparse.ArgumentParser(description="Run Flan-T5 inference.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input .parquet file"
    )
    parser.add_argument(
        "--output", type=str, required=False, help="Path to output .parquet file"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save progress every N steps (0 to disable)",
    )
    parser.add_argument(
        "--interim-save-path",
        type=str,
        default=None,
        help="Path to save interim results (optional)",
    )
    parser.add_argument(
        "--targets", type=str, default=None, help="Optional target string for labeling"
    )
    return parser


def decoder_args():
    parser = argparse.ArgumentParser(description="PMI Decoder")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the data file (parquet format)",
    )
    parser.add_argument(
        "--context-free-data",
        type=str,
        required=True,
        help="Path to the context-free data file (parquet format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to save the decoded output (parquet format)",
    )
    return parser


def evaluation_args():
    parser = argparse.ArgumentParser(description="Evaluation Arguments")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input CSV file for evaluation",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        required=False,
        help="Prompt template key for evaluation",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=False,
        help="Instruction key for evaluation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--decoding",
        type=str,
        required=True,
        help="greedy or pmi or aft",
    )

    return parser
