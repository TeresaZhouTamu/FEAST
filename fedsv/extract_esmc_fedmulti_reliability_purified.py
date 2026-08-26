import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import os
from utils.esmc_utils import get_esmc_layer_and_feature_dim, load_esmc_model, extract_esmc_features

def apply_orthogonal_rejection(sv, mu_neg, mu_test, dampening, is_hinge):
    """
    Removes the component of the steering vector (sv) that aligns with the 
    domain shift (mu_test - mu_neg).
    """
    if not is_hinge:
        return sv
        
    domain_diff = mu_test - mu_neg
    d_dot_d = torch.dot(domain_diff, domain_diff)
    
    if d_dot_d > 1e-8:
        projection = (torch.dot(sv, domain_diff) / d_dot_d) * domain_diff
        return sv - (dampening * projection)
    
    return sv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sv_paths', type=str, nargs='+', required=True, 
                        help="Space-separated paths to size-scaled client steering vectors (.pt files).")
    parser.add_argument('--data_path_test', type=str, required=True, 
                        help="CSV with Test sequences (The Target Domain).")
    parser.add_argument('--property', type=str, required=True, help="Property name.")

    # =========================== Settings =============================
    parser.add_argument('--num_data', type=int, default=None)
    parser.add_argument('--save_folder', type=str, default="saved_steering_vectors")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # toxicity:
    parser.add_argument('--dampening', type=float, default=0.5,
                        help="Strength of purification. (Use 0.0 for highly heterogeneous non-IID data).")
    parser.add_argument('--hinge_ratio', type=float, default=1.0,
                        help="Early layers to purify (0.2 = first 20% of layers).")
    
    parser.add_argument('--patch_alpha', type=float, default=0.5, 
                        help="Multiplier for the orthogonal auxiliary residual.")
    
    # immunogenicity:
    # parser.add_argument('--dampening', type=float, default=0.5,
    #                     help="Strength of purification. (Use 0.0 for highly heterogeneous non-IID data).")
    # parser.add_argument('--hinge_ratio', type=float, default=0.2,
    #                     help="Early layers to purify (0.2 = first 20% of layers).")
    
    # parser.add_argument('--patch_alpha', type=float, default=1.0, 
    #                     help="Multiplier for the orthogonal auxiliary residual.")
    parser.add_argument('--penalty_alpha', type=float, default=1.0,
                        help="Multiplier for disagreement penalty. Use 1.0 for IID, and 0.0 for non-IID.")
    parser.add_argument('--gate_threshold', type=float, default=0.15,
                        help="Minimum cosine similarity required to inject auxiliary features.")

    args = parser.parse_args()

    num_clients = len(args.sv_paths)
    print(f"Initializing fusion pipeline for {num_clients} clients...")

    # Load Pre-computed Size-Scaled Vectors for All Clients Dynamically
    clients_pos = []
    clients_neg = []
    client_names = []

    for path in args.sv_paths:
        name = os.path.basename(path).replace(".pt", "")
        client_names.append(name)
        print(f"Loading {name} components...")
        pos_all, neg_all = torch.load(path, map_location=args.device)
        clients_pos.append(pos_all)
        clients_neg.append(neg_all)

    # Load Model & Test Data
    n_layers, feature_dim = get_esmc_layer_and_feature_dim()
    model, tokenizer = load_esmc_model(device=args.device)
    hinge_layer = int(n_layers * args.hinge_ratio)

    df_test = pd.read_csv(args.data_path_test)
    test_seqs = df_test['sequence'].to_list()
    if args.num_data:
        test_seqs = test_seqs[:args.num_data]

    print(f"Extracting features for Test Domain manifold ({len(test_seqs)} seqs)...")
    test_repr = extract_esmc_features(test_seqs, model, tokenizer, n_layers)

    # --- Determine Global Anchor ---
    global_norms = []
    for idx in range(num_clients):
        total_norm = sum(torch.norm(clients_pos[idx][i].to(args.device) - clients_neg[idx][i].to(args.device)).item() for i in range(n_layers))
        global_norms.append(total_norm)
        
    global_anchor_idx = global_norms.index(max(global_norms))
    print(f"Global Anchor Selected: {client_names[global_anchor_idx]}")

    # Purification & Advanced N-Client Synergy Fusion
    print(f"Applying Purification & Global Anchored Fusion (Hinge: {hinge_layer})...")
    sv_combined_list = []
    sv_first_client_purified_list = [] 
    
    anchor_counts = {name: 0 for name in client_names}

    for i in range(n_layers):
        mu_test = test_repr[i].mean(dim=0).to(args.device)
        is_hinge = (i < hinge_layer)
        
        # --- Multi-Client Purification Loop ---
        purified_svs = []
        for idx in range(num_clients):
            sv_raw = clients_pos[idx][i].to(args.device) - clients_neg[idx][i].to(args.device)
            sv_prime = apply_orthogonal_rejection(
                sv_raw, clients_neg[idx][i].to(args.device), mu_test, args.dampening, is_hinge
            )
            purified_svs.append(sv_prime)
            
        sv_first_client_purified_list.append(purified_svs[0])

        # --- Global Anchoring Implementation ---
        sv_anchor = purified_svs[global_anchor_idx]
        sv_aux_list = [purified_svs[idx] for idx in range(num_clients) if idx != global_anchor_idx]
        
        anchor_counts[client_names[global_anchor_idx]] += 1
        sv_combined = sv_anchor.clone()
        anchor_dot = torch.dot(sv_anchor, sv_anchor) + 1e-8
        
        for sv_aux in sv_aux_list:
            cos_sim = F.cosine_similarity(sv_anchor, sv_aux, dim=0)
            
            # Only accept auxiliary features if there is meaningful alignment
            if cos_sim >= args.gate_threshold:
                # Extract ONLY the part of Aux that aligns with Anchor
                projection_aux_on_anchor = (torch.dot(sv_aux, sv_anchor) / anchor_dot) * sv_anchor
                
                # Accumulate shared features sequentially
                sv_combined = sv_combined + (args.patch_alpha * cos_sim * projection_aux_on_anchor)
            else:
                # Disagreement Penalty (Conflict)
                penalty = 1.0 - (args.penalty_alpha * torch.abs(cos_sim))
                penalty = max(penalty, 0.5) # Raised floor so the vector never collapses fully
                sv_combined = sv_combined * penalty
                
        current_norm = torch.norm(sv_combined) + 1e-8
        sv_combined = (sv_combined / current_norm) * torch.norm(sv_anchor)
        
        sv_combined_list.append(sv_combined)

    print(f"Fusion Complete. Anchor Distribution Profile: {anchor_counts}")

    # Save
    sv_combined_tensor = torch.stack(sv_combined_list).detach().cpu()
    # sv_first_client_tensor = torch.stack(sv_first_client_purified_list).detach().cpu()

    os.makedirs(args.save_folder, exist_ok=True)
    output_filename = f"ESMC_{args.property}_multi_client_reliable_synergy.pt"
    output_path = os.path.join(args.save_folder, output_filename)
    
    torch.save((sv_combined_tensor), output_path)
    print(f"Scalable synergy vectors saved to: {output_path}")