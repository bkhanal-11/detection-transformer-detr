import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is not None")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        assert tensor.dim() == 4, f"tensor.dim() = {tensor.dim()}"
        assert tensor.size(1) == self.num_pos_feats
        device, dtype = tensor.device, tensor.dtype
        _, _, H, W = tensor.shape

        # create coordinate tensors of shape (H, W)
        y_embed = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)
        x_embed = torch.arange(W, device=device, dtype=dtype).unsqueeze(0)
        y_embed = y_embed.repeat(1, W)
        x_embed = x_embed.repeat(H, 1)
        
        # normalize to [-1, 1]
        y_embed = (y_embed / ((H - 1) / 2.)) - 1
        x_embed = (x_embed / ((W - 1) / 2.)) - 1
        
        # apply scaling
        if self.scale is not None:
            y_embed = y_embed * self.scale
            x_embed = x_embed * self.scale
        
        # create tensor of shape (H, W, 2*num_pos_feats)
        pos = torch.cat([x_embed.unsqueeze(2), y_embed.unsqueeze(2)], dim=2).permute(2, 0, 1).unsqueeze(0).repeat(tensor.size(0), 1, 1, 1)
        
        # apply sine function
        pos = pos / self.temperature
        pos = torch.stack([pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()], dim=4).flatten(3)
        
        # normalize
        if self.normalize:
            pos = pos / pos.norm(p=2, dim=3, keepdim=True)
        
        return pos


def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells
    
    Arguments:
        decoder_token_ids -- (n, m) matrix
    
    Returns:
        mask -- (n, 1, m) binary tensor
    """
    # Create a matrix mask for the padding cells
    seq = 1 - torch.eq(decoder_token_ids, 0).float()
  
    # Add extra dimensions to add the padding to the attention logits
    return seq.unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(sequence_length):
    """
    Returns a lower triangular matrix filled with ones
    
    Arguments:
        sequence_length -- matrix size
    
    Returns:
        mask -- (size, size) tensor
    """
    # Create a lower triangular matrix filled with ones
    mask = torch.tril(torch.ones(sequence_length, sequence_length))
    
    return mask