import os
import sys
import time
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from argparse import ArgumentParser
import logging

from transformers import BertConfig, BertModel, BertTokenizer, AdamW

# ------------------------------
# Helper functions and loss
# ------------------------------
def set_all_seeds(seed):
    """Set all seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def split_rows(row):
    sep_indices = [i for i, x in enumerate(row['code']) if x == 'SEP']
    sep_indices = sep_indices[:-1]
    idx = np.random.choice(sep_indices) + 1 if sep_indices else len(row['code'])
    part1 = {
        'code': row['code'][:idx],
        'age': row['age'][:idx],
        'year': row['year'][:idx]
    }
    part2 = {
        'code': row['code'][idx:],
        'age': row['age'][idx:],
        'year': row['year'][idx:]
    }
    return part1, part2

def process_df_with_epochs(df, n_epochs=1):
    df_pre_list = []
    df_post_list = []
    for _ in range(n_epochs):
        set_all_seeds(seed + _)
        parts = df.apply(split_rows, axis=1)
        df1_data = [item[0] for item in parts]
        df2_data = [item[1] for item in parts]
        df_pre = pd.DataFrame(df1_data)
        df_post = pd.DataFrame(df2_data)
        non_list_columns = ['pracid', 'patid', 'dob', f'{args.disease}_date', 'enddate', 'label', 'row_num']
        df_pre[non_list_columns] = df[non_list_columns]
        df_post[non_list_columns] = df[non_list_columns]
        df_pre_list.append(df_pre)
        df_post_list.append(df_post)
    final_df_pre = pd.concat(df_pre_list)
    final_df_post = pd.concat(df_post_list)
    return final_df_pre, final_df_post

def calculate_prediction_strength(test_set, test_labels, training_centers):
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
        # Convert list of codes to a single string.
        codes = self.df.loc[idx, 'code']
        text = " ".join(codes) if isinstance(codes, list) else str(codes)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        # Squeeze to remove the batch dimension
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        return item

class PairedDataset(Dataset):
    """
    A paired dataset for contrastive learning.
    Expects two dataframes (pre and post splits) and returns a tuple of encoded items.
    """
    def __init__(self, df_pre, df_post, tokenizer, max_length=512):
        self.dataset_pre = EHRDataset(df_pre, tokenizer, max_length)
        self.dataset_post = EHRDataset(df_post, tokenizer, max_length)
        assert len(self.dataset_pre) == len(self.dataset_post), "Both datasets must be of equal length!"

    def __len__(self):
        return len(self.dataset_pre)

    def __getitem__(self, idx):
        return self.dataset_pre[idx], self.dataset_post[idx]

# ------------------------------
# Contrastive Loss (Multiple Negatives Ranking Loss)
# ------------------------------
class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale=20):
        super().__init__()
        self.scale = scale

    def forward(self, rep1, rep2):
        # Compute cosine similarity matrix scaled by self.scale
        sim_matrix = torch.matmul(rep1, rep2.T) * self.scale
        labels = torch.arange(rep1.size(0)).to(rep1.device)
        loss1 = F.cross_entropy(sim_matrix, labels)
        loss2 = F.cross_entropy(sim_matrix.T, labels)
        return (loss1 + loss2) / 2

# ------------------------------
# Model Initialization and Optimizer (replace with your own model)
# ------------------------------
def init_model_and_optimizer(device, lr=3e-5):
    config = BertConfig(
        vocab_size=30522,  # vocab size; adjust if needed
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        hidden_dropout_prob=0.15,
        attention_probs_dropout_prob=0.2,
    )
    model = BertModel(config)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    return model, optimizer

# ------------------------------
# Evaluation and Representation Generation
# ------------------------------
def generate_rep(batch, model, device):
    # For a given batch dictionary, move to device and compute BERT representations.
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    last_hidden_state = outputs.last_hidden_state
    mask = batch['attention_mask'].unsqueeze(-1).float()
    mean_pooled = torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
    return mean_pooled

def do_eval(dataloader, model, device):
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeds = generate_rep(batch, model, device)
            all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds, dim=0)

# ------------------------------
# Training Loop
# ------------------------------
def do_train(train_loader, eval_loader, model, optimizer, output_file_path, model_save_path, epochs=1000, patience=20, eval_steps=2000, scale=20, device='cpu'):
    step = 0
    loss_fct = MultipleNegativesRankingLoss(scale=scale)
    best_loss = float('inf')
    patience_counter = 0

    with open(output_file_path, 'w') as output_file:
        for epoch in range(epochs):
            output_file.write(f'Epoch {epoch + 1}/{epochs}\n')
            model.train()
            for batch in tqdm(train_loader, desc="Training"):
                optimizer.zero_grad()
                inputs_pre, inputs_post = batch
                rep_first = generate_rep(inputs_pre, model, device)
                rep_second = generate_rep(inputs_post, model, device)
                loss = loss_fct(rep_first, rep_second)
                loss.backward()
                optimizer.step()

                step += 1
                if step % 100 == 0:
                    output_file.write(f'Step {step}, Loss: {loss.item()}\n')
                    output_file.flush()

                if step > 1000 and step % eval_steps == 0:
                    model.eval()
                    with torch.no_grad():
                        eval_loss = 0.0
                        eval_steps_count = 0
                        for batch in tqdm(eval_loader, desc="Evaluating"):
                            inputs_pre, inputs_post = batch
                            rep_first = generate_rep(inputs_pre, model, device)
                            rep_second = generate_rep(inputs_post, model, device)
                            loss_eval = loss_fct(rep_first, rep_second)
                            eval_loss += loss_eval.item()
                            eval_steps_count += 1
                        avg_eval_loss = eval_loss / eval_steps_count
                        output_file.write(f'Validation Loss after {step} steps: {avg_eval_loss}\n')
                        if avg_eval_loss < best_loss:
                            best_loss = avg_eval_loss
                            torch.save(model.state_dict(), model_save_path)
                            output_file.write(f'Model saved at {model_save_path}\n')
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                output_file.write('Early stopping due to no improvement.\n')
                                return
                    model.train()

# ------------------------------
# Logging and Argument Parsing
# ------------------------------
parser = ArgumentParser()
parser.add_argument("--disease", type=str, default='HF')
parser.add_argument("--cohort_dir", type=str, default='HF_data')
parser.add_argument("--experiment_dir", type=str, default='HF') 
parser.add_argument("--model_name", type=str, default='bert_cl_b32')
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--scale", type=int, default=1)
parser.add_argument("--device", type=str, default='0')
args = parser.parse_args()

seed = args.seed
set_all_seeds(seed)

# Define paths (adjust these paths as needed)
COHORT_SAVE_PATH = "/path/to/your/cohort_dir"
MODEL_SAVE_PATH = "/path/to/save/models"
experiment_dir = os.path.join(MODEL_SAVE_PATH, args.experiment_dir)
os.makedirs(experiment_dir, exist_ok=True)
model_save_dir = os.path.join(experiment_dir, args.model_name)
os.makedirs(model_save_dir, exist_ok=True)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(model_save_dir, 'model.log'))
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info("Beginning contrastive training.")

# ------------------------------
# Data Loading and Preprocessing
# ------------------------------
cohort_path = os.path.join(COHORT_SAVE_PATH, args.cohort_dir)
ehr_data_path = os.path.join(cohort_path, 'EHR')
ehr_df_internal = pd.read_parquet(os.path.join(ehr_data_path, f'ehr_b4_{args.disease}_internal'))
ehr_df_external = pd.read_parquet(os.path.join(ehr_data_path, f'ehr_b4_{args.disease}_external'))

set_all_seeds(seed)
ehr_df_internal_pre, ehr_df_internal_post = process_df_with_epochs(ehr_df_internal)

ehr_df_internal = ehr_df_internal[
    (ehr_df_internal_post['code'].apply(len) > 0) &
    (ehr_df_internal_post['age'].apply(len) > 0) &
    (ehr_df_internal_post['year'].apply(len) > 0)
].reset_index(drop=True)
ehr_df_internal_pre = ehr_df_internal_pre[
    (ehr_df_internal_post['code'].apply(len) > 0) &
    (ehr_df_internal_post['age'].apply(len) > 0) &
    (ehr_df_internal_post['year'].apply(len) > 0)
].reset_index(drop=True)
ehr_df_internal_post = ehr_df_internal_post[
    (ehr_df_internal_post['code'].apply(len) > 0) &
    (ehr_df_internal_post['age'].apply(len) > 0) &
    (ehr_df_internal_post['year'].apply(len) > 0)
].reset_index(drop=True)

# ------------------------------
# Tokenizer for processing EHR data
# ------------------------------
# Make sure to replace 'model_name' with the appropriate tokenizer model name if needed.
tokenizer = BertTokenizer.from_pretrained('model_name')

# ------------------------------
# KFold Training and Evaluation
# ------------------------------
ps_threshold = 0.9
scale = args.scale
all_ps_scores = []
kf = KFold(n_splits=5, shuffle=True)
cluster_range = range(2, 10)  # Adjust as needed

def plot_ps_figure(ps_scores, fig_name='bertcl-k.pdf'):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(cluster_range, ps_scores, marker='o', linestyle='-', color='b', label='Prediction Strength')
    plt.axhline(y=ps_threshold, color='g', linestyle='--', label=f'Threshold = {ps_threshold}')
    plt.title('Prediction Strength for Different k Values')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Prediction Strength')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(global_params['output_dir'], fig_name))
    plt.close()

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
global_params = {
    'batch_size': args.batch_size,
    'output_dir': model_save_dir,
    'max_len_seq': 512,
    'max_age': 110,
    'age_year': False,
    'age_symbol': None,
    'min_visit': 5,
    'yearOn': True,
    'device': device,
}

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(ehr_df_internal)):
    set_all_seeds(seed)
    # Create paired datasets for contrastive training using the pre and post splits
    paired_train_dataset = PairedDataset(
        ehr_df_internal_pre.loc[train_idx].reset_index(drop=True),
        ehr_df_internal_post.loc[train_idx].reset_index(drop=True),
        tokenizer,
        max_length=global_params['max_len_seq']
    )
    paired_train_loader = DataLoader(paired_train_dataset, batch_size=global_params['batch_size'], shuffle=True)
    
    paired_test_dataset = PairedDataset(
        ehr_df_internal_pre.loc[test_idx].reset_index(drop=True),
        ehr_df_internal_post.loc[test_idx].reset_index(drop=True),
        tokenizer,
        max_length=global_params['max_len_seq']
    )
    paired_test_loader = DataLoader(paired_test_dataset, batch_size=global_params['batch_size'] * 16, shuffle=False)
    
    # Create evaluation datasets (single view) from the original internal data
    eval_train_dataset = EHRDataset(ehr_df_internal.loc[train_idx].reset_index(drop=True), tokenizer, max_length=global_params['max_len_seq'])
    eval_train_loader = DataLoader(eval_train_dataset, batch_size=global_params['batch_size'], shuffle=False)
    eval_test_dataset = EHRDataset(ehr_df_internal.loc[test_idx].reset_index(drop=True), tokenizer, max_length=global_params['max_len_seq'])
    eval_test_loader = DataLoader(eval_test_dataset, batch_size=global_params['batch_size'], shuffle=False)
    
    torch.cuda.empty_cache()
    model, optimizer = init_model_and_optimizer(device, lr=3e-5)
    
    output_file_path = os.path.join(global_params['output_dir'], f'MNR_scale_{scale}_fold{fold_idx}.txt')
    model_save_path = os.path.join(global_params['output_dir'], f'MNR_scale_{scale}_fold{fold_idx}.pt')
    
    if not os.path.exists(model_save_path):
        do_train(paired_train_loader, paired_test_loader, model, optimizer, output_file_path, model_save_path,
                 epochs=10, patience=5, eval_steps=500, scale=scale, device=device)
    logger.info('Completed training, loading the best model.')
    model.load_state_dict(torch.load(model_save_path))
    
    # Evaluate to obtain embeddings
    X_train = do_eval(eval_train_loader, model, device)
    X_test = do_eval(eval_test_loader, model, device)
    
    ps_k = []
    for k in cluster_range:
        trainingCluster = KMeans(n_clusters=k, max_iter=50, n_init=10, random_state=seed).fit(X_train)
        testCluster = KMeans(n_clusters=k, max_iter=50, n_init=10, random_state=seed).fit(X_test)
        train_labels = trainingCluster.predict(X_test)
        ps_k.append(calculate_prediction_strength(X_test, testCluster.labels_, trainingCluster.cluster_centers_))
    logger.info(f"fold_idx: {fold_idx}, fold_ps: {ps_k}")
    logger.info(f"fold_idx: {fold_idx}, max ps cluster index: {np.argmax(ps_k)}, max ps values: {np.max(ps_k)}")
    plot_ps_figure(ps_k, fig_name=f'bertcl-k-fold{fold_idx}.pdf')
    all_ps_scores.append(ps_k)

ps_scores = np.mean(all_ps_scores, axis=0)
logger.info(ps_scores)

optimal_k_ps = None
for idx, score in reversed(list(enumerate(ps_scores))):
    if score >= ps_threshold:
        optimal_k_ps = cluster_range[idx]
        break

if optimal_k_ps is not None:
    optimal_ps_score = ps_scores[optimal_k_ps - min(cluster_range)]
    logger.info(f"Prediction Strength Score for optimal k ({optimal_k_ps}): {optimal_ps_score}")
    logger.info(f"Optimal k based on prediction strength: {optimal_k_ps}")

    best_fold_ps_at_optimal_k = -1
    best_fold_idx = -1
    for fold_idx, ps_scores_for_fold in enumerate(all_ps_scores):
        if ps_scores_for_fold[optimal_k_ps - min(cluster_range)] > best_fold_ps_at_optimal_k:
            best_fold_ps_at_optimal_k = ps_scores_for_fold[optimal_k_ps - min(cluster_range)]
            best_fold_idx = fold_idx
    if best_fold_idx >= 0:
        logger.info(f"The best fold for PS at k={optimal_k_ps} was fold #{best_fold_idx + 1} with PS={best_fold_ps_at_optimal_k}")
else:
    logger.info("No optimal k found with PS above the threshold.")

plot_ps_figure(ps_scores, fig_name='bertcl-k.pdf')
