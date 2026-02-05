# Comparative Evaluation of Sequence-to-Sequence Models (T5, BART, DistilBART)

This repository contains a unified Python script designed to compare the performance and inference speed of popular Sequence-to-Sequence models (T5, BART, DistilBART) across different Natural Language Processing (NLP) tasks.

The script is built to be **hardware-agnostic**, capable of running efficiently on **TPU, GPU, and CPU** environments using dynamic device detection with PyTorch XLA and standard CUDA/CPU setup.

## ✨ Features

* **Unified Pipeline:** A single script handles dependency installation, model loading, data preprocessing, metric calculation, and visualization.
* **Multi-Task Evaluation:** Simultaneously evaluates models on multiple tasks, including:
    * **Classification:** SST2 (Single-Sentence) and MRPC (Sentence Pair).
    * **Summarization:** CNN/DailyMail.
* **Comprehensive Metrics:** Calculates key performance indicators based on the task type:
    * **Classification:** Accuracy.
    * **Summarization:** ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore (F1).
* **Performance Benchmarking:** Records and reports **Inference Time (seconds per sample)** for performance comparison across different hardware.
* **Cross-Device Support:** Automatically prioritizes and configures for **TPU** (via PyTorch XLA), **GPU** (via CUDA), or **CPU** execution.

## ⚙️ Installation and Setup

The script requires a standard Python environment. If running in a Google Colab or Kaggle notebook, ensure you select the appropriate runtime type (GPU or TPU) before starting.

### Prerequisites

* Python 3.8+

### Dependencies

Run the first code block in the script to install all necessary dependencies, including `torch_xla` for optional TPU support.

```bash
# Example of dependencies installed by the script:
# pip install transformers datasets evaluate pandas scikit-learn bert_score rouge_score seaborn matplotlib
# pip install torch_xla
