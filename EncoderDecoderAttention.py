import torch
import torch.nn as nn

class EncoderDecoderAttention(nn.Module):
    def __init__(self, input_dimensions):
        super().__init__()
        self.query_layer = nn.Linear(input_dimensions, input_dimensions)
        self.key_layer = nn.Linear(input_dimensions, input_dimensions)
        self.value_layer = nn.Linear(input_dimensions, input_dimensions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = self.value_layer(value)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)
        output = torch.matmul(attention_weights, V)
        return output


