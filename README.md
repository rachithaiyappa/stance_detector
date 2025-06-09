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

## Dataset:
The `data/` directory contains the text (tweets) for which stance is to be detected along with the target of interest.
The `csv` file has the following columns   
- `ID`: identifier for the tweet
- `Target`: Target of interest 
- `Tweet`: Tweet (text) 
- `Stance`: Ground truth stance label to use for downstream evaluation. 

Example csv file:

```csv
ID,Target,Tweet,Stance
10001,Atheism,He who exalts himself shall      be humbled; and he who humbles himself shall be exalted.Matt 23:12.     #SemST,AGAINST
10002,Atheism,"RT @prayerbullets: I remove Nehushtan -previous moves of God that have become idols, from the high places -2 Kings 18:4 #SemST",AGAINST
```

We have included the data from which all the results of the paper can be reproduced, in `data/`
### [P-Stance](https://aclanthology.org/2021.findings-acl.208/)

- `pstance_bernie.csv`: Tweets for the target Bernie Sanders
- `pstance_biden.csv`: Tweets for the target Joe Biden
- `pstance_bernie.csv`: Tweets for the target Donald Trump

### [SemEval 2016 Task 6](https://aclanthology.org/L16-1623/)

- `semeval_taskA.csv`: Tweets for the targets belonging to Task 6A 
    - Atheism
    - Hillary Clinton
    - Climate Change in a Real Concern
    - Feminist Movement
    - Legalization of Abortion.
- `semeval_taskB.csv`: Tweets for the target belonging to Task 6B
    - Donald Trump.

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
  title={Zero-Shot Stance Detection in Practice: Insights on Training, Prompting, and Decoding with a Capable Lightweight LLM},
  author={Aiyappa, Rachith and Senthilmani, Shruthi and An, Jisun and Kwak, Haewoon and Ahn, Yong-Yeol},
  journal={arXiv preprint arXiv:2403.00236},
  year={2024}
}
```


## For questions or issues, please open an issue on GitHub.