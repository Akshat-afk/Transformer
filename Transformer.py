import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, vocabulary_size, number_of_embeddings, sequence_len, input_dimensions):
        super().__init__()
        self.encoder = Encoder(
            vocabulary_size=vocabulary_size,
            number_of_embeddings=number_of_embeddings,
            sequence_len=sequence_len,
            input_dimensions=input_dimensions
            )
        self.fcl = nn.Sequential(
            nn.Linear(input_dimensions,1280),
            nn.ReLU(),
            nn.Linear(1280, input_dimensions),
            nn.Softmax(dim=-1)
        )
        self.decoder = Decoder(
            vocabulary_size=vocabulary_size,
            number_of_embeddings=number_of_embeddings,
            sequence_len=sequence_len,
            input_dimensions=input_dimensions
            )

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(x, encoder_output)
        return decoder_output

