import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import ResNet50
from transformer import Transformer
from utils import PositionalEmbedding

class DETR(nn.Module):
    def __init__(self, num_classes, num_encoder_layers, num_decoder_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_seq_len):
        super().__init__()

        # backbone network
        self.backbone = ResNet50()
        self.backbone.layer4 = nn.Identity()

        # positional encoding
        self.position_embedding = PositionalEmbedding(d_model)

        # transformer encoder
        self.transformer = Transformer(num_encoder_layers, num_decoder_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_seq_len)

        # object queries
        self.query_embed = nn.Embedding(num_classes, d_model)

        # class and box prediction heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

    def forward(self, x):
        # backbone network feature extraction
        backbone_features = self.backbone(x)

        # flatten and permute feature maps
        flattened_features = backbone_features.flatten(2).permute(2, 0, 1)

        # add positional encoding
        encoded_features = self.position_embedding(flattened_features)

        # transformer 
        decoded_features = self.transformer(self.input_proj(backbone_features), self.query_embed.weight, encoded_features)[0]
        
        # class and box predictions
        class_predictions = self.class_embed(decoded_features).sigmoid()
        bbox_predictions = self.bbox_embed(decoded_features).sigmoid()

        return class_predictions, bbox_predictions

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x