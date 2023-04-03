import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class FullyConnected(nn.Module):
    def __init__(self, embedding_dim, fully_connected_dim):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, fully_connected_dim)
        self.relu = nn.GELU()
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

    def forward(self, x, mask=None):
        """
        Forward pass for the Encoder Layer
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, d_model)
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            encoder_layer_out -- Tensor of shape (batch_size, input_seq_len, d_model)
        """
        # calculate Self-Attention using Multi-Head Attention
        mha_output, _ = self.mha(x, x, x, mask)  # Self attention (batch_size, input_seq_len, d_model)

        # skip connection
        # apply layer normalization on sum of the input and the attention output to get the output of the multi-head attention layer
        skip_x_attention = self.layernorm1(x + mha_output)

        # pass the output of the multi-head attention layer through a ffn
        ffn_output = self.ffn(skip_x_attention)

        # apply dropout layer to ffn output during training
        ffn_output = self.dropout_ffn(ffn_output)

        # apply layer normalization on sum of the output from multi-head attention (skip connection) and ffn output to get the output of the encoder layer
        encoder_layer_out = self.layernorm2(skip_x_attention + ffn_output)

        return encoder_layer_out

class Encoder(nn.Module):
    """
    The encoder is composed by a stack of identical layers (EncoderLayers).
    """
    def __init__(self, num_layers, num_heads, d_model, fully_connected_dim,
                 input_vocab_size, maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.enc_layers = nn.ModuleList([EncoderLayer(num_heads, d_model, fully_connected_dim,
                                                       dropout_rate=dropout_rate, layernorm_eps=layernorm_eps)
                                          for _ in range(num_layers)])
        
    def forward(self, x, mask=None):
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
            x = self.enc_layers[i](x, mask)
        
        encoder_out = x
        
        return encoder_out
    
