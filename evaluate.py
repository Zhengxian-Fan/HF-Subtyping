import os
import sys
import time
import random
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from tqdm.notebook import tqdm

# Import Hugging Face components
from transformers import BertModel, BertTokenizer

from argparse import ArgumentParser
import logging

# ------------------------------
# Helper Functions
# ------------------------------
def set_all_seeds(seed):
    """Set all seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def calculate_prediction_strength(test_set, test_labels, training_centers):
    # Precompute distances for all points to training centers
    distances = np.linalg.norm(test_set[:, np.newaxis] - training_centers, axis=2)
    closest_centers = np.argmin(distances, axis=1)
    
    prediction_strengths = []
    for cluster in range(training_centers.shape[0]):
        indices = np.where(test_labels == cluster)[0]
        cluster_size = len(indices)
        if cluster_size <= 1:
            prediction_strengths.append(float('inf'))
        else:
            cluster_closest_centers = closest_centers[indices]
            matching_pairs = np.sum(np.triu((cluster_closest_centers[:, None] == cluster_closest_centers), 1))
            total_pairs = cluster_size * (cluster_size - 1) / 2.0
            prediction_strengths.append(matching_pairs / total_pairs)
    return min(prediction_strengths)

# ------------------------------
# EHR Dataset
# ------------------------------
class EHRDataset(Dataset):
    """
    A simple dataset class that expects a DataFrame with a 'code' column.
    Each entry in 'code' should be a list of tokens which will be joined into a single string.
    """
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        codes = self.df.loc[idx, 'code']
        text = " ".join(codes) if isinstance(codes, list) else str(codes)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        # Remove the batch dimension
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        return item

# ------------------------------
# Evaluation Function
# ------------------------------
def do_eval(dataloader, model, device):
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            last_hidden_state = outputs.last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).float()
            mean_pooled = torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            all_embeds.append(mean_pooled.cpu())
    return torch.cat(all_embeds, dim=0)

# ------------------------------
# t-SNE Plot Function
# ------------------------------
def tsne_plot(embeddings, labels, k, name='internal'):
    from MulticoreTSNE import MulticoreTSNE as TSNE  # Import if needed
    tsne = TSNE(n_components=2, n_iter=300, random_state=seed)
    X_test_2D = tsne.fit_transform(embeddings[:10000])
    plt.figure(figsize=(5,5), dpi=100)
    colors = sns.color_palette("deep", k)
    for cluster_id in range(1, k+1):
        cluster_mask = (labels[:10000] == cluster_id)
        cluster_data = X_test_2D[cluster_mask]
        plt.scatter(
            cluster_data[:, 0], cluster_data[:, 1],
            alpha=1.0, color=colors[cluster_id-1],
            s=5,
            label=f'Cluster {cluster_id}'
        )
    plt.title(f't-SNE visualization for {k} clusters')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large', markerscale=2.5)
    plt.savefig(os.path.join(results_save_dir, f'tsne-{name}-{k}.pdf'))
    plt.show()

# ------------------------------
# Main Script
# ------------------------------
parser = ArgumentParser()
parser.add_argument("--disease", type=str, default='HF')
parser.add_argument("--cohort_dir", type=str, default='HF_data')
parser.add_argument("--experient_dir", type=str, default='HF')
parser.add_argument("--model_name", type=str, default='bert_random_b32')
parser.add_argument("--scale", type=int, default=1)
parser.add_argument("--k", type=int, default=12)
parser.add_argument("--fold_idx", type=int, default=2)
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--device", type=str, default='0')
# New argument: path to the previously trained model checkpoint.
parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
args = parser.parse_args()

seed = args.seed
set_all_seeds(seed)

# Define directories (adjust these paths as necessary)
COHORT_SAVE_PATH = "/path/to/your/cohorts"
MODEL_SAVE_PATH = "/path/to/save/models"
results_save_dir = os.path.join(MODEL_SAVE_PATH, args.experient_dir, args.model_name, f'{args.disease}-{args.k}')
os.makedirs(results_save_dir, exist_ok=True)

# Load your EHR DataFrames (assuming parquet format)
ehr_data_path = os.path.join(COHORT_SAVE_PATH, args.cohort_dir, 'EHR')
ehr_df_internal = pd.read_parquet(os.path.join(ehr_data_path, f'ehr_b4_{args.disease}_internal'))
ehr_df_external = pd.read_parquet(os.path.join(ehr_data_path, f'ehr_b4_{args.disease}_external'))

# Filter out rows with empty 'code' lists
ehr_df_internal = ehr_df_internal[ehr_df_internal['code'].apply(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)
ehr_df_external = ehr_df_external[ehr_df_external['code'].apply(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)

# Tokenizer for processing EHR data.
# Replace 'model_name' with the appropriate tokenizer if needed.
tokenizer = BertTokenizer.from_pretrained(args.model_checkpoint)

# Create datasets
internal_dataset = EHRDataset(ehr_df_internal, tokenizer, max_length=512)
external_dataset = EHRDataset(ehr_df_external, tokenizer, max_length=512)

# Create DataLoaders
internal_loader = DataLoader(internal_dataset, batch_size=args.batch_size * 4, shuffle=True)
external_loader = DataLoader(external_dataset, batch_size=args.batch_size * 4, shuffle=False)

# Load the trained model checkpoint
device = f'cuda:{args.device}' if torch.cuda.is_available() and len(args.device) == 1 else 'cpu'
model = BertModel.from_pretrained(args.model_checkpoint)
model.to(device)
model.eval()

# Evaluate and save embeddings
internal_embeddings = do_eval(internal_loader, model, device)
external_embeddings = do_eval(external_loader, model, device)

torch.save(internal_embeddings, os.path.join(results_save_dir, 'eval_internal.pt'))
torch.save(external_embeddings, os.path.join(results_save_dir, 'eval_external.pt'))

# Cluster the internal embeddings using KMeans
optimal_k = args.k
kmeans_model = KMeans(n_clusters=optimal_k, max_iter=50, n_init=10, random_state=seed)
train_labels = kmeans_model.fit_predict(internal_embeddings)
external_labels = kmeans_model.predict(external_embeddings)
# Optionally adjust labels to start from 1
train_labels += 1
external_labels += 1

# t-SNE visualization
print('Internal validation t-SNE plot')
tsne_plot(internal_embeddings, train_labels, optimal_k, name=f'{args.disease}-internal')
print('External validation t-SNE plot')
tsne_plot(external_embeddings, external_labels, optimal_k, name=f'{args.disease}-external')

# Save labels back to DataFrames and store
ehr_df_internal['label'] = train_labels
ehr_df_external['label'] = external_labels
ehr_df_internal.to_parquet(os.path.join(results_save_dir, f'ehr_b4_{args.disease}_internal_with_label.parquet'), compression='snappy')
ehr_df_external.to_parquet(os.path.join(results_save_dir, f'ehr_b4_{args.disease}_external_with_label.parquet'), compression='snappy')
