# Stance Detector

This repository contains code to reproduce the results from the paper:  
[Zero-Shot Stance Detection in Practice: Insights on Training, Prompting, and Decoding with a Capable Lightweight LLM](https://arxiv.org/abs/2403.00236)

## Features

- **Pre-process Tweets** (Optional):  Clean and normalize tweet data for downstream tasks.
- **Prompt Construction:** Build prompts for LLMs using various templates and instructions.  
  See [`src/prompt_config.py`](src/prompt_config.py) for available prompt templates and instructions.
- **Prompt Perplexity Measurement:** Evaluate prompt perplexity using an encoder-decoder model (FlanT5-XXL).
- **Stance Labeling:**
  - Greedy decoding with FlanT5-XXL
  - PMI decoding
  - AfT decoding
- **Evaluation:** Assess stance labels produced by different decoding strategies.

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/rachithaiyappa/stance_detector.git
    ```

2. Switch to the directory 
    ```sh
    cd stance_detector
    ```

3. Install the package and its dependencies in a virtual environent
    ```sh
    ./setup.sh
    ```

## Requirements
- [FlanT5-XXL](https://huggingface.co/google/flan-t5-xxl) (requires GPUs with at least 80GB memory; tested on NVIDIA A100)  
- See `project.toml` for Python version and dependencies.


## If you use this code, please cite our paper:
TBD but for now 

```
@article{aiyappa2024benchmarking,
  title={Benchmarking zero-shot stance detection with FlanT5-XXL: Insights from training data, prompting, and decoding strategies into its near-SoTA performance},
  author={Aiyappa, Rachith and Senthilmani, Shruthi and An, Jisun and Kwak, Haewoon and Ahn, Yong-Yeol},
  journal={arXiv preprint arXiv:2403.00236},
  year={2024}
}
```


## For questions or issues, please open an issue on GitHub.