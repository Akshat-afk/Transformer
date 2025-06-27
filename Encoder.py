import torch
import torch.nn as nn
import SelfAttention
import WordEmbedding
import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, vocabulary_size, number_of_embeddings, sequence_len, input_dimensions):
        super().__init__()
        self.word_embedding = WordEmbedding.WordEmbeddingNN(vocabulary_size,number_of_embeddings)
        self.self_attentionNN = SelfAttention.SelfAttention(input_dimensions)
        self.register_buffer("positional_encodings", PositionalEncoding.positionalEncoding(sequence_len, number_of_embeddings))

    def forward(self, x):
        word_embeddings = self.word_embedding(x)
        positional_encodings = self.positional_encodings[:x.size(1), :] + word_embeddings
        self_attention_values = self.self_attentionNN(positional_encodings)
        output = self_attention_values + positional_encodings
        return output

