import torch
import torch.nn as nn
import SelfAttention
import WordEmbedding
import PositionalEncoding
import EncoderDecoderAttention

class Decoder(nn.Module):
    def __init__(self, vocabulary_size, number_of_embeddings, sequence_len, input_dimensions):
        super().__init__()
        self.word_embedding = WordEmbedding.WordEmbeddingNN(vocabulary_size,number_of_embeddings)
        self.self_attention = SelfAttention.SelfAttention(input_dimensions)
        self.encoder_decoder_attention = EncoderDecoderAttention.EncoderDecoderAttention(input_dimensions)
        self.fcl = nn.Linear(input_dimensions, vocabulary_size)
        self.register_buffer("positional_encodings", PositionalEncoding.positionalEncoding(sequence_len, number_of_embeddings))

    def forward(self, x, encoder_output):
        word_embeddings = self.word_embedding(x)
        positional_encodings = self.positional_encodings[:x.size(1), :] + word_embeddings
        self_attention_values = self.self_attention(positional_encodings)
        residual_connection = self_attention_values + positional_encodings
        encoder_decoder_output = self.encoder_decoder_attention(query=residual_connection, key = encoder_output, value = encoder_output)
        output = self.fcl(encoder_decoder_output)
        return output

