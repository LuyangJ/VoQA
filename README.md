<h2 align="center"> <a href="http://arxiv.org/abs/2505.14227">VoQA: Visual-only Question Answering</a><h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2505.14227-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2505.14227)[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/LuyangJ/VoQA/blob/main/LICENSE)

[ English | [中文](./README_zh.md) ]

##

We propose **Visual-only Question Answering (VoQA)**, a novel multimodal task in which questions are visually embedded within images, without any accompanying textual input. Here is the difference between **Traditional VQA Task** and **VoQA Task**:

<p align="center">
    <img src="https://github.com/LuyangJ/VoQA/raw/main/assets/VoQA-task.png" width="600">
</p>

**VoQA Task** poses challenges for existing large vision-language models (LVLMs), which show notable performance drops even with carefully designed prompts.

<p align="center">
    <img src="https://github.com/LuyangJ/VoQA/raw/main/assets/VoQA-Zero-shot-performance.png" width="750">
</p>

To bridge this gap, we introduce **Guided Response Triggering Supervised Fine-tuning (GRT-SFT)**, a structured fine-tuning strategy that guides the model to perform step-by-step reasoning purely based on visual input, significantly improving model performance. Here is the difference between **Baseline-SFT**, **QA-SFT** and **GRT-SFT**:

<p align="center">
    <img src="https://github.com/LuyangJ/VoQA/raw/main/assets/GRT-SFT.png" width="750">
</p>

## Project Overview

VoQA Benchmark is a comprehensive benchmark for Visual-only Question Answering (VoQA) that provides a unified evaluation framework for both open-source and closed-source models. This project integrates evaluation logic for various datasets and models, allowing for **efficient batch evaluation through simple parameter modifications**.

## Key Features

* Evaluation of open-source models on the VoQA dataset
* API-based evaluation of closed-source models on the VoQA dataset
* Evaluation of open-source models on traditional VQA datasets
* API-based evaluation of closed-source models on traditional VQA datasets
* Response filtering and accuracy calculation based on existing inference results
* Evaluation of GRT-SFT strategy models for Question Recognition Accuracy (QRA)

## Quick Start

### 1. Dataset Preparation

**Step 1:** Download the evaluation datasets from 🤗Hugging Face: [AJN-AI/VoQA](https://huggingface.co/datasets/AJN-AI/VoQA)

Scripts for only downloading the test split are available at Hugging Face.

**Step 2:** Modify the `EVAL_DIR` parameter in the main scripts to point to your evaluation datasets root directory.

### 2. Model and Environment Setup (Open-source Models Only)

The project includes the following pre-configured models:

* TinyLLaVA\_Factory: TinyLLaVA-Phi-2-SigLIP-3.1B, TinyLLaVA-Qwen2-0.5B-SigLIP, TinyLLaVA-Qwen2.5-3B-SigLIP
* LLaVA: llava-v1.5-7b, llava-1.5-7b-hf
* Qwen: Qwen2.5-VL-3B-Instruct, Qwen2-VL-2B
* InternVL: InternVL2\_5-1B
* DeepSeek\_VL2: deepseek-vl2-tiny
* LAVIS\_xgen\_mm: xgen-mm-phi3-mini-instruct-interleave-r-v1.5 (BLIP-3)

Each model series has its corresponding conda environment configuration. Take `TinyLLaVA_Factory` as an example:

First, switch the directory to `TinyLLaVA_Factory`:

```Shell
cd models/TinyLLaVA_Factory
```

Then execute the commands in `README.md`:

```Shell
conda create -n tinyllava_factory python=3.10 -y
conda activate tinyllava_factory
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install flash-attn --no-build-isolation
```

### 3. Evaluation Process

After preparing for the datasets, models, and conda environments, you just need to modify a few parameters in the main script. The descriptions of each parameter have been provided in each script and can be modified by referring to the existing examples.

#### Open-source Model Evaluation

* Traditional VQA Datasets: Modify `scripts_for_traditional_vqa/eval_traditional_vqa_main.sh`

```Bash
bash scripts_for_traditional_vqa/eval_traditional_vqa_main.sh
```

* VoQA Concatenation Dataset: Modify `scripts_for_voqa/eval_concatenation_for_zero_shot.sh`

```Bash
bash scripts_for_voqa/eval_concatenation_for_zero_shot.sh
```

* VoQA Watermark Dataset: Modify `scripts_for_voqa/eval_watermark_for_zero_shot.sh`

```Bash
bash scripts_for_voqa/eval_watermark_for_zero_shot.sh
```

#### Closed-source Model Evaluation

* Traditional VQA Datasets: Modify `scripts_for_traditional_vqa_api/eval_for_traditional_vqa.sh`

```Bash
bash scripts_for_traditional_vqa_api/eval_for_traditional_vqa.sh
```

* VoQA Concatenation Dataset: Modify `scripts_for_voqa_api/eval_concatenation_for_voqa.sh`

```Bash
bash scripts_for_voqa_api/eval_concatenation_for_voqa.sh
```

* VoQA Watermark Dataset: Modify `scripts_for_voqa_api/eval_watermark_for_voqa.sh`

```Bash
bash scripts_for_voqa_api/eval_watermark_for_voqa.sh
```

## Project Structure

```Plain
VoQA-code/
├── eval/                                       # Core evaluation logic
│   ├── api_for_submit.py                       # Submit a single request using the api
│   ├── convert_gqa_for_eval.py                 # Evaluation functions for GQA
│   ├── convert_vqav2_for_submission.py         # Evaluation functions for VQAv2
│   ├── eval_for_api.py                         # The main function for evaluating the close-source models
│   ├── eval_main.py                            # The main function for evaluating the open-source models
│   ├── eval_pope.py                            # Evaluation functions for POPE
│   ├── eval_science_qa.py                      # Evaluation functions for SQA
│   ├── eval_textvqa.py                         # Evaluation functions for TextVQA
│   ├── load_models.py                          # Scripts for controlling the loading of different open-source models
│   ├── m4c_evaluator.py                        # Evaluation functions for TextVQA and VQAv2
│   ├── models_inference.py                     # Scripts for controlling different open-source models for reasoning
│   └── process_answer.py                       # Functions for Response filtering
├── models/                                     # Model project folders
├── scripts_for_response_filtering/             # Response filtering scripts
├── scripts_for_result_analysis/                # GRT-SFT result analysis
├── scripts_for_traditional_vqa/                # Traditional VQA evaluation scripts
├── scripts_for_traditional_vqa_api/            # Traditional VQA API evaluation scripts
├── scripts_for_voqa/                           # VoQA evaluation scripts
└── scripts_for_voqa_api/                       # VoQA API evaluation scripts
```

## Adding New Models

To add a new model for evaluation, follow these steps:

**Step 1:** Create a new model folder

**Step 2:** Configure the conda environment for your model

**Step 3:** Add new branches in `eval/load_models.py` and `eval/models_inference.py`

**Step 4:** Implement model loading and inference functions

## Additional functions

In addition to the basic evaluation code, we have also implemented two additional functions: **Response Filtering** and **GRT-SFT Result Analysis**.

### Response Filtering

If you have saved inference results and want to modify the answer processing logic, update the logic in `eval/process_answer.py`. Then modify parameters in:

* `scripts_for_response_filtering/eval_concatenation_with_filter_main.sh` for VoQA concatenation dataset

```Bash
bash scripts_for_response_filtering/eval_concatenation_with_filter_main.sh
```

* `scripts_for_response_filtering/eval_watermark_with_filter_main.sh` for VoQA watermark dataset

```Bash
bash scripts_for_response_filtering/eval_watermark_with_filter_main.sh
```

### GRT-SFT Result Analysis

To analyze the Question Recognition Accuracy (QRA) of models fine-tuned with the GRT-SFT strategy, use `scripts_for_result_analysis/analyse_for_grt_main.sh` and modify its parameters as needed.

```Bash
bash scripts_for_result_analysis/analyse_for_grt_main.sh
```

## ❤️ Acknowledgement
* Our whole evaluation codebase is built upon the [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) project. Great work!
* `./models` part of our evaluation codebase is built upon the projects as follows: [TinyLLaVA_Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory), [LLaVA](https://github.com/haotian-liu/LLaVA), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [InternVL](https://github.com/OpenGVLab/InternVL/tree/v2.5), [DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2), [BLIP-3](https://github.com/salesforce/LAVIS/tree/xgen-mm). Great works!
* Our project uses data from the [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V) project. Great work!

## License

See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.
For models and datasets, please refer to the original resource page and follow the corresponding License.

## Citation

If you find our paper, code or datasets helpful in your research, please consider giving a star :star: and kindly cite as:
```bibtex
@article{jiang2025voqa,
  title={VoQA: Visual-only Question Answering},
  author={Jiang, Luyang and An, Jianing and Luo, Jie and Wu, Wenjun and Huang, Lei},
  journal={arXiv preprint arXiv:2505.14227},
  year={2025}
}
```

