from typing import Union, Tuple, List
from attr import dataclass
try:
    from flash_attn.bert_padding import pad_input, unpad_input  # type:ignore

    is_flash_attn_available = True
except ImportError:
    pad_input = None
    unpad_input = None
    is_flash_attn_available = False
import torch
from esm.utils.structure.affine3d import Affine3D

@dataclass
class ESMCOutput:
    sequence_logits: torch.Tensor
    embeddings: torch.Tensor | None
    hidden_states: torch.Tensor | None

def steering_forward(
    self,
    x: torch.Tensor,
    sequence_id: Union[torch.Tensor, None] = None,
    affine: Union[Affine3D, None] = None,
    affine_mask: Union[torch.Tensor, None] = None,
    chain_id: Union[torch.Tensor, None] = None,
    steering_vectors: Union[List[torch.Tensor], None] = None, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Steering Forward pass of the TransformerStack.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).
        sequence_id (torch.Tensor): The sequence ID tensor of shape (batch_size, sequence_length).
        affine (Affine3D, None]): The affine transformation tensor or None.
        affine_mask (torch.Tensor, None]): The affine mask tensor or None.
        chain_id (torch.Tensor): The protein chain tensor of shape (batch_size, sequence_length).
            Only used in geometric attention.

    Returns:
        post_norm: The output tensor of shape (batch_size, sequence_length, d_model).
        pre_norm: The embedding of shape (batch_size, sequence_length, d_model).
    """
    # *batch_dims, _ = x.shape
    # if chain_id is None:
    #     chain_id = torch.ones(size=batch_dims, dtype=torch.int64, device=x.device)
    
    # for l, block in enumerate(self.blocks):
    #     x = block(x, sequence_id, affine, affine_mask, chain_id)

    #     if steering_vectors is not None:
    #         add_x = steering_vectors[l]
    #         new_x = x + add_x
    #         new_x_norm = torch.norm(new_x, p=2, dim=-1, keepdim=True).detach()
    #         x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).detach()
    #         x = new_x * (x_norm / new_x_norm) 

    # return self.norm(x), x
    *batch_dims, _ = x.shape
    if chain_id is None:
        chain_id = torch.ones(size=batch_dims, dtype=torch.int64, device=x.device)
    hiddens = []
    for l, block in enumerate(self.blocks):
        x = block(x, sequence_id, affine, affine_mask, chain_id)

        if steering_vectors is not None:
            add_x = steering_vectors[l]
            new_x = x + add_x
            new_x_norm = torch.norm(new_x, p=2, dim=-1, keepdim=True).detach()
            x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).detach()
            x = new_x * (x_norm / new_x_norm)

        hiddens.append(x)

    return self.norm(x), x, hiddens

def esmc_steering_forward(
    self,
    sequence_tokens: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
    steering_vectors: Union[List[torch.Tensor], None] = None, 
) -> torch.Tensor:
    # """
    # Performs forward pass through the ESMC model. Check utils to see how to tokenize inputs from raw data.

    # Args:
    #     sequence_tokens (torch.Tensor, optional): The amino acid tokens.
    #     sequence_id (torch.Tensor, optional): The sequence ID.

    # Returns:
    #     ESMCOutput: The output of the ESMC model.

    # """
    if sequence_id is None:
        # For EMSC, a boolean mask is created in place of sequence_id if not specified.
        sequence_id = sequence_tokens != self.tokenizer.pad_token_id

    x = self.embed(sequence_tokens)

    B, L = x.shape[:2]

    # If sequence_id looks like a mask.
    if self._use_flash_attn:
        assert (
            sequence_id.dtype == torch.bool
        ), "sequence_id must be a boolean mask if Flash Attention is used"
        assert sequence_id.shape == (B, L)
        assert unpad_input is not None
        x, indices, *_ = unpad_input(  # type: ignore
            x, sequence_id
        )
    else:
        indices = None

    x, _, hiddens = self.transformer.steering_forward(x, sequence_id=sequence_id, steering_vectors=steering_vectors)

    if self._use_flash_attn:
        assert indices is not None
        assert pad_input is not None
        x = pad_input(x, indices, B, L)  # Back to [B, L, D]
        hiddens = [
            # Back to [[B, L, D], ...]
            pad_input(h, indices, B, L)
            for h in hiddens
        ]

    # Stack hidden states into a [n_layers, B, L, D] matrix.
    hiddens = torch.stack(hiddens, dim=0)  # type: ignore

    sequence_logits = self.sequence_head(x)
    output = ESMCOutput(
        sequence_logits=sequence_logits, embeddings=x, hidden_states=hiddens
    )
    return output