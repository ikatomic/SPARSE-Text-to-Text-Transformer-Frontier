````markdown
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
````

## 🚀 Usage

The entire evaluation is executed by running the script sequentially.

### 1\. Configure Hardware

The script automatically detects the best available hardware:

1.  **TPU:** If PyTorch XLA is successfully imported and a TPU core is found (`xm.xla_device()`), the model will run on TPU.
2.  **GPU:** If no TPU is found but a CUDA device is available (`torch.cuda.is_available()`), the model will run on GPU.
3.  **CPU:** If neither TPU nor GPU is available, the model will run on CPU.

### 2\. Define Models and Tasks

The models and tasks are defined in the `MODEL_NAMES` and `TASK_CONFIGS` dictionaries:

| Model Name | Type | Notes |
| :--- | :--- | :--- |
| `t5-small` | Seq2Seq | Requires task-specific prefixes (e.g., "summarize: ") |
| `facebook/bart-base` | Seq2Seq | |
| `sshleifer/distilbart-cnn-12-6` | Seq2Seq | Smaller, faster BART for summarization |

| Task Key | Dataset | Type | Metric |
| :--- | :--- | :--- | :--- |
| `sst2` | GLUE/SST-2 | Classification | Accuracy |
| `mrpc` | GLUE/MRPC | Classification | Accuracy |
| `cnn_dailymail` | CNN/DailyMail | Summarization | ROUGE, BERTScore |

### 3\. Run the Evaluation

The core of the script runs a nested loop: `Model` -\> `Task`.

The `evaluate_model` function performs the following steps for each model-task pair:

1.  Loads a small, shuffled sample of the dataset (currently **50 samples** for fast benchmarking).
2.  Preprocesses and tokenizes the data, adding T5 prefixes where necessary.
3.  Runs the model generation loop, moving data to the detected device (CPU/GPU/TPU).
4.  Decodes the predictions and calculates the relevant metrics (Accuracy, ROUGE, BERTScore).
5.  Records the inference time.

### 4\. Review Results

The script outputs a final `pandas.DataFrame` summarizing all results:

| Model | Task | Accuracy | ROUGE-1 F1 | BERTScore F1 | Inference Time (s/sample) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| t5-small | sst2 | 0.880 | N/A | N/A | 0.051 |
| facebook/bart-base | cnn\_dailymail | N/A | 0.385 | 0.901 | 0.125 |
| ... | ... | ... | ... | ... | ... |

Additionally, the script generates a series of bar charts using `seaborn` to visualize the performance across all metrics and models.

## 🛠️ Key Technical Implementations

The script includes several key fixes and improvements over the original notebook:

  * **Robust Preprocessing:** Handles the specific input format requirements of T5 for both single-sentence and sentence-pair GLUE tasks, and summarization tasks.
  * **Dynamic Device Switching:** The `evaluate_model` function includes logic to set the execution device:
    ```python
    if XLA_AVAILABLE and ...:
        device = xm.xla_device()  # TPU
    elif torch.cuda.is_available():
        device = "cuda"        # GPU
    else:
        device = "cpu"         # CPU
    ```
  * **Accurate TPU Timing:** The use of `xm.mark_step()` within the inference loop and `xm.wait_device_ops()` after the loop ensures precise, synchronous timing of the model's execution when running on a TPU.

<!-- end list -->

```
```
