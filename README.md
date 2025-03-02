# EHR Contrastive Learning and Subtyping

This repository contains code to apply contrastive learning on Electronic Health Record (EHR) data and generate patient subtypes. The approach leverages a Transformer-based model fine-tuned using a contrastive loss on sequential EHR data. The code computes patient embeddings, clusters them using KMeans, and calculates the optimal k using prediction strength.

> **Note:** Due to data privacy restrictions, the original vocabulary and model files used in our research cannot be shared. In this repository, a Hugging Face BERT model is used as a placeholder. Please replace it with your own model and vocabulary if necessary.

## Overview

- **finetune.py**  
  Fine-tunes a Transformer-based model on EHR data using contrastive learning.

- **evaluate.py**  
  Generates patient subtypes by computing embeddings with the trained model, clustering them via KMeans, and producing visualizations (e.g., t-SNE plots).

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- pandas
- matplotlib
- seaborn
- MulticoreTSNE

Install the dependencies with:

```bash
pip install torch transformers scikit-learn pandas matplotlib seaborn MulticoreTSNE

## Usage

### Fine-Tuning the Model

Run the following command to fine-tune the model on your EHR data:

```bash
python finetune.py --disease HF --cohort_dir HF_data --experiment_dir HF --model_name cl_maskage_b16 --batch_size 16 --seed 12345 --device 1
```

#### Parameter Explanations:

- `--disease HF`: Specifies the disease label for training.
- `--cohort_dir HF_data`: Directory containing the HF cohort data.
- `--experiment_dir HF`: Directory to save the experiment outputs, including checkpoints and logs.
- `--model_name HF_model`: The designated name for the trained model.
- `--batch_size 16`: Batch size used during training.
- `--seed 12345`: Seed value for reproducibility.
- `--device 1`: GPU device identifier for training.

### Evaluating the Trained Model and Generating Subtypes

After training, generate patient subtypes using the evaluation script:

```bash
python evaluate.py --disease HF --cohort_dir HF_data --model_name HF_model --experiment_dir HF --k 6 --fold_idx 4 --device 0 
```

#### Parameter Explanations:

- `--disease HF`: Specifies the disease label for evaluation.
- `--cohort_dir HF_data`: Directory containing the HF cohort data.
- `--experiment_dir HF`: Directory where evaluation results will be saved.
- `--model_name HF_model`: The designated name for the trained model.
- `--k 6`: Number of clusters/subtypes to be generated.
- `--fold_idx 4`: Fold index for cross-validation (if applicable).
- `--device 0`: GPU device identifier for evaluation.


