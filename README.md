# Xerostomia Detection via Fine-Tuning Medical VLMs

This repository contains a specialized pipeline for fine-tuning Vision-Language Models (VLMs) to detect **Xerostomia** (dry mouth) from intraoral images. By leveraging the `google/medgemma-1.5-4b-it` model and state-of-the-art optimization techniques, this project achieves high-accuracy diagnostic classification through patient-level nested cross-validation.

## Key Features

- **Specialized Medical VLM**: Fine-tuned on `MedGemma 1.5`, a model specifically pre-trained for medical contexts.
- **Robust Evaluation**: Implementation of **Nested Cross-Validation** (5-fold outer, 4-fold inner) to ensure generalization and unbiased performance estimation.
- **Automated Hyperparameter Tuning**: Integrated with **Optuna** to optimize Rank (r), Alpha, Learning Rate, and Target Modules.
- **Efficient Pipeline**: Uses **Unsloth** for 2x faster training and 70% less memory usage, and pre-cached HuggingFace datasets for rapid patient-level fold splitting.
- **Zero-Shot Baseline**: Includes scripts for benchmarking performance against non-fine-tuned models.

---

## Repository Structure

```text
.
├── data-processing/
│   ├── aug.py             # Data augmentation using Albumentations (5x)
│   └── cache_dataset.py   # Pre-processes images into cached HF Datasets
├── nested-cv/
│   ├── config.py          # Global configurations and model paths
│   ├── main.py            # Entry point for Nested Cross-Validation & Optuna
│   ├── modeling.py        # Unsloth/PEFT model setup and training logic
│   ├── utils.py           # GPU management and patient-level fold logic
│   └── experiment_results.txt # Detailed logs from previous runs
├── llm-classification.py   # Zero-shot baseline classification script
└── README.md
```

---

## Setup & Installation

### 1. Environment Setup

> [!IMPORTANT]
> **CUDA MANDATORY!!** This project requires an NVIDIA GPU with CUDA 12.1+ for VLM fine-tuning.

This project uses `uv` for lightning-fast dependency management. To set up your environment:

```bash
# Install dependencies and create a virtual environment
uv sync

# Activate the environment
source .venv/bin/activate
```

Alternatively, you can install manually using the `pyproject.toml`:

```bash
pip install .
```

### 2. Data Preparation

Place your raw patient images in a `./raw` directory. Folders starting with `X` are treated as positive cases (Xerostomia), others as negative.

---

## Usage

### Phase 1: Data Pre-processing

First, augment the dataset to improve model robustness:

```bash
python data-processing/aug.py
```

Then, cache the images into patient-specific datasets for efficient loading:

```bash
python data-processing/cache_dataset.py
```

### Phase 2: Zero-Shot Baseline

To evaluate the model's performance without any fine-tuning:

```bash
python llm-classification.py
```

### Phase 3: Fine-Tuning & Evaluation

Run the nested cross-validation pipeline. This will perform hyperparameter optimization for each outer fold and save the best parameters.

```bash
python nested-cv/main.py
```

---

## Methodology

### Nested Cross-Validation

To avoid overfitting and provide a realistic estimate of the model's performance on new patients:

1. **Outer Loop (5 Folds)**: Splits patients into 5 groups. One group is held out as the final test set in each iteration.
2. **Inner Loop (4 Folds)**: For every outer fold, the training data is split again into 4 folds for **Optuna** to find the best hyperparameters.
3. **Final Evaluation**: The model is trained on the full outer training set using the best parameters and evaluated on the held-out test patients.

### Hyperparameter Search Space

- **Learning Rate**: $5 \times 10^{-6}$ to $5 \times 10^{-4}$ (log scale)
- **LoRA Rank (r)**: [8, 16, 32, 64]
- **LoRA Alpha**: [16, 32, 64]
- **Target Modules**: Various combinations of attention and MLP layers.

---

## Technologies Used

- **Model**: [Google MedGemma 1.5 4B IT](https://huggingface.co/google/medgemma-1.5-4b-it)
- **Fine-Tuning**: [Unsloth](https://unsloth.ai/) & [PEFT (LoRA)](https://github.com/huggingface/peft)
- **Optimization**: [Optuna](https://optuna.org/)
- **Data**: [Albumentations](https://albumentations.ai/), [HuggingFace Datasets](https://huggingface.co/docs/datasets/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
