import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import os
import numpy as np
import math
from utils.esmc_utils import get_esmc_layer_and_feature_dim, load_esmc_model, extract_esmc_features

def compute_weighted_mean(features, weights, n_layers):
    """
    Computes a weighted average across the sample dimension for each layer.
    weights: Tensor normalized to sum to 1.
    features: List of tensors 
    """
    weighted_means = []
    w = weights.unsqueeze(1).to(features[0].device)
    
    for layer_idx in range(n_layers):
        layer_feat = features[layer_idx] 
        # Weighted sum
        weighted_feat = (layer_feat * w).sum(dim=0)
        weighted_means.append(weighted_feat)
        
    return torch.stack(weighted_means)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', type=str, nargs='+', required=True, 
                        help="CSV files with: sequence, label, confidence, variability, correctness")
    parser.add_argument('--property', type=str, required=True)
    parser.add_argument('--num_data', type=int, default=None)
    parser.add_argument('--save_folder', type=str, default="saved_steering_vectors")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Initialize Model
    n_layers, feature_dim = get_esmc_layer_and_feature_dim()
    try:
        model, tokenizer = load_esmc_model(device=args.device)
    except TypeError:
        model, tokenizer = load_esmc_model()
        model = model.to(args.device)

    all_pos_vectors = []
    all_neg_vectors = []

    # Process each dataset
    for path in args.data_paths:
        if not os.path.exists(path):
            continue

        print(f"\n--- Processing {path} with Size-Normalized Cartography Weights ---")
        df = pd.read_csv(path)
        seq_col = 'aa_seq' if 'aa_seq' in df.columns else 'sequence'
        
        # Weighting schema for immunogenicty:
        df = df[df['correctness'] >= 0.4].copy()
        df['raw_weight'] = df['confidence'] 

        # # Weighting schema for toxicity:
        # var_std = df['variability'].std() if df['variability'].std() > 0 else 0.1
        # df['raw_weight'] = df["correctness"] * df['confidence'] * np.exp(-df['variability'] / var_std)
        n_pos = len(df[df['label'] == 1])
        n_neg = len(df[df['label'] == 0])

        if n_pos == 0 or n_neg == 0:
            print(f"Skipping {path}: One or both classes are empty after cleaning filters.")
            continue

        effective_n = math.sqrt(n_pos * n_neg)
        size_scalar = math.sqrt(effective_n)
        print(f"Dataset Scale -> Pos: {n_pos}, Neg: {n_neg} | Effective N: {effective_n:.2f} | Scaling Scalar: {size_scalar:.4f}")

        for label in [1, 0]:
            sub_df = df[df['label'] == label].copy()
            if args.num_data is not None:
                sub_df = sub_df.head(args.num_data)

            if sub_df.empty:
                continue

            # Normalize 
            total_w = sub_df['raw_weight'].sum()
            if total_w > 0:
                sub_df['norm_weight'] = sub_df['raw_weight'] / total_w
            else:
                sub_df['norm_weight'] = 1.0 / len(sub_df)

            seqs = sub_df[seq_col].tolist()
            weights_tensor = torch.tensor(sub_df['norm_weight'].values, dtype=torch.float32)

            print(f"Extracting features for Label {label} ({len(seqs)} samples)...")
            reprs = extract_esmc_features(seqs, model, tokenizer, n_layers)
            weighted_centroid = compute_weighted_mean(reprs, weights_tensor, n_layers)
            scaled_centroid = weighted_centroid * size_scalar
            
            if label == 1:
                all_pos_vectors.append(scaled_centroid.detach().cpu())
            else:
                all_neg_vectors.append(scaled_centroid.detach().cpu())

    if not all_pos_vectors or not all_neg_vectors:
        raise ValueError("Could not find both positive and negative samples across valid inputs.")

    print("\n--- Finalizing Size-Normalized & Cartography Weighted Steering Vectors ---")
    final_pos_steering = torch.stack(all_pos_vectors).sum(dim=0)
    final_neg_steering = torch.stack(all_neg_vectors).sum(dim=0)

    os.makedirs(args.save_folder, exist_ok=True)
    output_path = f"{args.save_folder}/ESMC_{args.property}_reliable_steering_vectors.pt"
    
    torch.save((final_pos_steering, final_neg_steering), output_path)
    print(f"Success! Size-scaled aggregated vectors saved to: {output_path}")