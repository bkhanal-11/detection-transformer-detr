import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import PositionalEmbedding

class FullyConnected(nn.Module):
    def __init__(self, embedding_dim, fully_connected_dim):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, fully_connected_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fully_connected_dim, embedding_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        
        return self.fc2(x)

class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        self.scale = torch.nn.Parameter(torch.ones(self.hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(self.hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        
        return self.scale * normalized + self.bias

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout_rate=0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.dropout = nn.Dropout(dropout_rate)

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)

        self.W_o = nn.Linear(self.head_dim * self.num_heads, self.d_model)

    def forward(self, queries, keys, values, mask=None):
        batch_size = queries.size(0)

        # Linear Transformation
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)
        
        # Reshaping Tensors
        queries = queries.view(batch_size, -1, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim)
        values = values.view(batch_size, -1, self.num_heads, self.head_dim)

        # Calculate dot products between the queries and the keys
        matmul_qk = torch.einsum('bqhd,bkhd->bhqk', [queries, keys])

        dk = keys.size()[-1]
        scaled_attention_logits = matmul_qk / (dk ** 0.5)

        if mask is not None:
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask==0, float('-1e20')) 
        
        # Apply softmax function to obtain attention weights
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # Apply dropout to the attention weights
        attention_weights_dropout = self.dropout(attention_weights)

        context = torch.einsum('bhqv,bvhd->bqhd', [attention_weights_dropout, values])

        context = context.reshape(batch_size, -1, self.num_heads*self.head_dim)

        # Concatenate heads and apply final linear transformation
        output = self.W_o(context)

        return output


class EncoderLayer(nn.Module):
    """
    The encoder layer is composed of a multi-head self-attention mechanism,
    followed by a simple, position-wise fully connected feed-forward network. 
    This architecture includes a residual connection around each of the two 
    sub-layers, followed by layer normalization.
    """
    def __init__(self, num_heads, d_model, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      d_model=d_model,
                                      dropout_rate=dropout_rate)

        self.ffn = FullyConnected(embedding_dim=d_model,
                                  fully_connected_dim=fully_connected_dim)

        self.layernorm1 = LayerNorm(d_model, eps=layernorm_eps)
        self.layernorm2 = LayerNorm(d_model, eps=layernorm_eps)

        self.dropout_ffn = nn.Dropout(dropout_rate)
    
    def with_pos_embed(self, x, pos=None):
        return x if pos is None else x + pos

    def forward(self, x, mask=None, pos=None):
        """
        Forward pass for the Encoder Layer
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, d_model)
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            encoder_layer_out -- Tensor of shape (batch_size, input_seq_len, d_model)
        """
        x_2 = self.layernorm1(x)
        
        q = k = self.with_pos_embed(x_2, pos)

        mha_output = self.mha(q, k, x, mask)[0]  

        skip_x_attention = x + self.dropout_ffn(mha_output)
        
        skip_x_attention = self.layernorm2(skip_x_attention)

        ffn_output = self.ffn(skip_x_attention)
        
        ffn_output = self.dropout_ffn(ffn_output)

        encoder_layer_out = skip_x_attention + ffn_output

        return encoder_layer_out

class Encoder(nn.Module):
    """
    The encoder is composed by a stack of identical layers (EncoderLayers).
    """
    def __init__(self, num_layers, num_heads, d_model, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.enc_layers = nn.ModuleList([EncoderLayer(num_heads, d_model, fully_connected_dim,
                                                       dropout_rate=dropout_rate, layernorm_eps=layernorm_eps)
                                          for _ in range(num_layers)])
        
    def forward(self, x, mask=None, pos=None):
        """
        Forward pass for the Encoder
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len)
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            encoder_out -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """       
        # Pass the input through each encoder layer
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, pos)
        
        encoder_out = x
        
        return encoder_out
    
class DecoderLayer(nn.Module):
    """
    The decoder layer is composed by an masked multi-head self-attention mechanism,
    followed by a multi-head attention mechanism to the output of the encoder and a 
    simple, position-wise fully connected feed-forward network. This architecture 
    includes residual connections around all of the three sub-layers, followed by 
    layer normalization.
    """
    def __init__(self, num_heads, d_model, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        self.masked_mha = MultiHeadAttention(num_heads=num_heads,
                                             d_model=d_model,
                                             dropout_rate=dropout_rate)

        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      d_model=d_model,
                                      dropout_rate=dropout_rate)

        self.ffn = FullyConnected(embedding_dim=d_model,
                                  fully_connected_dim=fully_connected_dim)

        self.layernorm1 = LayerNorm(d_model, eps=layernorm_eps)
        self.layernorm2 = LayerNorm(d_model, eps=layernorm_eps)
        self.layernorm3 = LayerNorm(d_model, eps=layernorm_eps)

        self.dropout_ffn = nn.Dropout(dropout_rate)
    
    def with_pos_embed(self, x, pos=None):
        return x if pos is None else x + pos

    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None, pos=None, query_pos=None):
        x_2 = self.layernorm1(x)
        
        q = k = self.with_pos_embed(x_2, pos)
        # masked multi-head self-attention
        mha_output1 = self.masked_mha(q, k, x, look_ahead_mask)[0] # (batch_size, target_seq_len, fully_connected_dim)

        # skip connection
        skip_x_attention1 = x + self.dropout_ffn(mha_output1)
        skip_x_attention1 = self.layernorm2(skip_x_attention1)

        # multi-head attention on encoder output
        mha_output2 = self.mha(self.with_pos_embed(skip_x_attention1, query_pos), 
                               self.with_pos_embed(encoder_output, pos), encoder_output, padding_mask)[0] # (batch_size, target_seq_len, fully_connected_dim)

        skip_x_attention2 = x + self.dropout_ffn(mha_output2)
        skip_x_attention2 = self.layernorm3(skip_x_attention2)

        ffn_output = self.ffn(skip_x_attention2)

        ffn_output = self.dropout_ffn(ffn_output)

        decoder_layer_out = skip_x_attention2 + ffn_output

        return decoder_layer_out

class Decoder(nn.Module):
    """
    The Decoder consists of N layers of DecoderLayer.
    """
    def __init__(self, num_layers, num_heads, d_model, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(num_heads=num_heads,
                         d_model=d_model,
                         fully_connected_dim=fully_connected_dim,
                         dropout_rate=dropout_rate,
                         layernorm_eps=layernorm_eps)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None, pos = None, query_pos=None):
        """
        Forward pass for the Decoder.
        
        Arguments:
            x -- Tensor of shape (batch_size, target_seq_len)
            enc_output -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
            look_ahead_mask -- Boolean mask for the target sequence
            padding_mask -- Boolean mask for the input sequence
            
        Returns:
            decoder_output -- Tensor of shape (batch_size, target_seq_len, embedding_dim)
            attention_weights -- Dictionary of attention weights for each decoder layer
        """
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, enc_output, look_ahead_mask, padding_mask, pos, query_pos)
        
        return x.unsqueeze(0)

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers=6, num_decoder_layers=6, d_model=512, num_heads=8, dff=2048, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_encoder_layers, num_heads, d_model, dff, dropout_rate)

        self.decoder = Decoder(num_decoder_layers, num_heads, d_model, dff, dropout_rate)
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        target = torch.zeros_like(query_embed)
        encoded = self.encoder(src, mask=mask, pos=pos_embed)
        
        output = self.decoder(target, encoded, padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        
        return output.transpose(1, 2), encoded.permute(1, 2, 0).view(bs, c, h, w)
    