import torch

# def steering_forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False, steering_vectors=None):           
#  NEW FLEXIBLE SIGNATURE
def steering_forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False, steering_vectors=None, **kwargs): 
    if return_contacts:
        need_head_weights = True

    assert tokens.ndim == 2
    padding_mask = tokens.eq(self.padding_idx)  # B, T

    x = self.embed_scale * self.embed_tokens(tokens)

    if padding_mask is not None:
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

    repr_layers = set(repr_layers)
    hidden_representations = {}
    if 0 in repr_layers:
        hidden_representations[0] = x

    if need_head_weights:
        attn_weights = []

    # (B, T, E) => (T, B, E)
    x = x.transpose(0, 1)

    if not padding_mask.any():
        padding_mask = None

    for layer_idx, layer in enumerate(self.layers):
        x, attn = layer(
            x,
            self_attn_padding_mask=padding_mask,
            need_head_weights=need_head_weights,
        )
        if steering_vectors is not None:
            add_x = steering_vectors[layer_idx]
            new_x = x + add_x
            new_x_norm = torch.norm(new_x, p=2, dim=-1, keepdim=True).detach()
            x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).detach()
            x = new_x * (x_norm / new_x_norm)

        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        if need_head_weights:
            # (H, B, T, T) => (B, H, T, T)
            attn_weights.append(attn.transpose(1, 0))

    x = self.emb_layer_norm_after(x)
    x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

    # last hidden representation should have layer norm applied
    if (layer_idx + 1) in repr_layers:
        hidden_representations[layer_idx + 1] = x
    x = self.lm_head(x)

    result = {"logits": x, "representations": hidden_representations}
    if need_head_weights:
        # attentions: B x L x H x T x T
        attentions = torch.stack(attn_weights, 1)
        if padding_mask is not None:
            attention_mask = 1 - padding_mask.type_as(attentions)
            attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            attentions = attentions * attention_mask[:, None, None, :, :]
        result["attentions"] = attentions
        if return_contacts:
            contacts = self.contact_head(tokens, attentions)
            result["contacts"] = contacts

    return result

# import torch

# def steering_forward(
#     self, 
#     tokens, 
#     repr_layers=[], 
#     need_head_weights=False, 
#     return_contacts=False, 
#     steering_vectors=None, 
#     **kwargs
# ):            
#     if return_contacts:
#         need_head_weights = True

#     assert tokens.ndim == 2
    
#     # 1. Grab Padding Token from Hugging Face Config
#     pad_idx = self.config.pad_token_id if hasattr(self, 'config') else 1
#     padding_mask = tokens.eq(pad_idx)  # B, T

#     # 2. Extract Hidden Size and Route Embeddings through Hugging Face layout
#     hidden_size = self.config.hidden_size if hasattr(self, 'config') else 320
#     embed_scale = hidden_size ** 0.5
    
#     # Pull word embeddings out of the HF EsmEmbeddings block
#     x = embed_scale * self.esm.embeddings.word_embeddings(tokens)

#     if padding_mask is not None:
#         x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

#     repr_layers = set(repr_layers)
#     hidden_representations = {}
#     if 0 in repr_layers:
#         hidden_representations[0] = x

#     if need_head_weights:
#         attn_weights = []

#     # (B, T, E) => (T, B, E) to remain compatible with your internal loop logic
#     x = x.transpose(0, 1)

#     if not padding_mask.any():
#         padding_mask = None

#     # 3. Route layers dynamically using Hugging Face's layout (.esm.encoder.layer)
#     layers_list = self.esm.encoder.layer if hasattr(self, 'esm') else self.layers
    
#     # Prepare expected attention mask shape for Hugging Face layer modules if mask is active
#     hf_attention_mask = None
#     if padding_mask is not None:
#         hf_attention_mask = (1.0 - (1 - padding_mask.long()).unsqueeze(1).unsqueeze(2)) * -10000.0
#         hf_attention_mask = hf_attention_mask.type_as(x)

#     # 💡 FIX HERE: Force pure inference optimization for attention tracking matrices
#     with torch.inference_mode():
#         for layer_idx, layer in enumerate(layers_list):
#             layer_outputs = layer(hidden_states=x.transpose(0, 1), attention_mask=hf_attention_mask)
#             x = layer_outputs[0].transpose(0, 1) # Turn back into (T, B, E)
#             attn = layer_outputs[1] if len(layer_outputs) > 1 else None

#             # --- Custom Steering Vector Math Loop ---
#             if steering_vectors is not None and layer_idx < len(steering_vectors):
#                 add_x = steering_vectors[layer_idx].unsqueeze(1).type_as(x)
#                 new_x = x + add_x
                
#                 new_x_norm = torch.norm(new_x, p=2, dim=-1, keepdim=True)
#                 x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
                
#                 x = new_x * (x_norm / (new_x_norm + 1e-8))

#             if (layer_idx + 1) in repr_layers:
#                 hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            
#         if need_head_weights and attn is not None:
#             attn_weights.append(attn)

#     # 4. Optional Post-Transformer Layer Normalization Routing
#     if hasattr(self.esm, 'contact_head') and hasattr(self.esm.contact_head, 'layer_norm'):
#         x = x.transpose(0, 1)
#         x = self.esm.contact_head.layer_norm(x)
#         x = x.transpose(0, 1)

#     x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

#     if (layer_idx + 1) in repr_layers:
#         hidden_representations[layer_idx + 1] = x
        
#     # 5. Project final hidden states through the LM classification head
#     x = self.lm_head(x)

#     # Pack results into dictionary structure
#     result = {"logits": x, "representations": hidden_representations}
    
#     if need_head_weights and len(attn_weights) > 0:
#         result["attentions"] = torch.stack(attn_weights, 1)
#         if return_contacts and hasattr(self, 'contact_head'):
#             result["contacts"] = self.contact_head(tokens, result["attentions"])

#     # 6. 💡 COMPATIBILITY & MEMORY WRAPPER
#     class ModelOutputWrapper:
#         def __init__(self, d):
#             # Detaching here completely cuts the cord to previous transformer layer histories
#             self.logits = d.get("logits").detach()
#             self.hidden_states = [h.detach() for h in d.get("representations", {}).values()]
            
#         def __getitem__(self, key):
#             return self.__dict__[key]

#     return ModelOutputWrapper(result)