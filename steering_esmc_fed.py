import argparse
import pandas as pd
import numpy as np
from itertools import islice
from tqdm import tqdm
import types
import torch
import torch.nn.functional as F
import os
from esm.utils.constants.esm3 import SEQUENCE_MASK_TOKEN
from utils.esmc_utils import decode_sequence, load_esmc_model, pred_tokens, get_tokenwise_representations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, 
                        help="Path to CSV file containing 'sequence' and 'label' columns (Test Set).")
    parser.add_argument('--optimal_layer', type=int, required=True, 
                        help="The specific layer index used to identify mutation sites (e.g., 3).")

    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    parser.add_argument('--property', type=str, required=True, help="Property name (e.g., 'immunog').")
    parser.add_argument('--output_file', type=str, default=None, help="Path to save the optimized sequences.")
    parser.add_argument('--alpha', type=float, default=2.0, help="Steering strength coefficient")
    parser.add_argument('--sv_from', type=str, default="saved_steering_vectors", help="Folder path where steering vectors are saved.")

    parser.add_argument('--n', type=int, default=1000, help="Maximal number of sequences to optimize.")
    parser.add_argument('--round', type=int, default=1, help="Number of optimization rounds.")
    parser.add_argument('--T', type=int, default=1, help="Number of mutation sites per round.")
    
    args = parser.parse_args() 

    print(f"Loading sequences from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    if 'label' in df.columns:
        neg_df = df[df['label'] == 0]
        org_seqs = neg_df['sequence'].to_list()
    else:
        org_seqs = df['sequence'].to_list()
    
    print(f"Filtered to {len(org_seqs)} sequences for optimization.")
    if len(org_seqs) == 0:
        print("Warning: No sequences found. Nothing to optimize.")
        exit(0)
    
    model, tokenizer = load_esmc_model(device=args.device)
    try:
        from module.steerable_esmc import steering_forward, esmc_steering_forward
        model.transformer.steering_forward = types.MethodType(steering_forward, model.transformer)
        model.steering_forward = types.MethodType(esmc_steering_forward, model)
    except ImportError:
        print("Error: Could not import steering methods from module.steerable_esmc.")
        exit(1)

    print(f"Loading purified steering vectors from: {args.sv_from}")
    
    if not os.path.exists(args.sv_from):
        raise FileNotFoundError(f"Steering vector file not found at: {args.sv_from}")

    sv_combined = torch.load(args.sv_from, map_location=args.device)
    
    steering_vectors = (sv_combined * args.alpha).to(args.device)
    layer_scoring_vec = sv_combined[args.optimal_layer].clone().to(args.device)

    new_seqs = [[] for _ in range(args.round)]
    print(f"Starting optimization using Optimal Layer: {args.optimal_layer}")

    with torch.no_grad():
        for seq in tqdm(islice(org_seqs, args.n), total=min(len(org_seqs), args.n)):
            raw_ids = tokenizer.encode(seq, add_special_tokens=False)
            bos_id = getattr(tokenizer, 'bos_token_id', 0) or 0
            eos_id = getattr(tokenizer, 'eos_token_id', 2) or 2
            
            token_ids = [bos_id] + raw_ids + [eos_id]
            seq_token = torch.tensor(token_ids, dtype=torch.int64, device=args.device)
            
            prev_seq_token = seq_token.clone()
            prev_mut_sites = set()

            for r in range(args.round):
                full_features = get_tokenwise_representations(
                    prev_seq_token.unsqueeze(0), 
                    torch.ones_like(prev_seq_token).unsqueeze(0), 
                    model
                )
                layer_features = full_features[0, :, args.optimal_layer, :]

                valid_features = layer_features[1:-1]
                related_score = F.cosine_similarity(valid_features, layer_scoring_vec.unsqueeze(0), dim=-1).cpu().numpy()

                sorted_indices = np.argsort(related_score)
                mut_sites = []
                for idx in sorted_indices:
                    if len(mut_sites) >= args.T:
                        break
                    if idx not in prev_mut_sites:
                        mut_sites.append(idx)
                
                prev_mut_sites.update(mut_sites)
                mut_sites_tensor = torch.LongTensor(mut_sites).to(args.device) + 1

                masked_seq = prev_seq_token.clone()
                masked_seq[mut_sites_tensor] = SEQUENCE_MASK_TOKEN
                
                new_seq_token = pred_tokens(
                    masked_seq, 
                    model, 
                    steering_vectors, 
                    original_prediction=prev_seq_token, 
                    temperature=0.0
                )
                
                prev_seq_token = masked_seq.clone()
                prev_seq_token[mut_sites_tensor] = new_seq_token[mut_sites_tensor]
                new_seqs[r].append(decode_sequence(prev_seq_token, tokenizer))
                
    last_round_idx = args.round - 1
    final_seqs = new_seqs[last_round_idx]
    final_epochs = [args.round] * len(final_seqs)

    res_df = pd.DataFrame({'sequence': final_seqs, 'epoch': final_epochs})
    
    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
        res_df.to_csv(args.output_file, index=False)
        print(f'Generated {len(res_df)} sequences (Round {args.round} only) saved to: {args.output_file}')
    else:
        print("Optimization complete (no output file specified).")