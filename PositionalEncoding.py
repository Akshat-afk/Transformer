import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def positionalEncoding(sequence_len, number_of_embeddings):
    positional_encodings = np.zeros((sequence_len, number_of_embeddings))
    for pos in range(sequence_len):
        for i in range(0, number_of_embeddings, 2):
            angle = pos / np.power(10000, (2 * i)/number_of_embeddings)
            positional_encodings[pos, i] = np.sin(angle)
            if i + 1 < number_of_embeddings:
                positional_encodings[pos, i + 1] = np.cos(angle)
    return torch.tensor(positional_encodings, dtype=torch.float32).to(device)

def testing():
    positionalEncoding(3,2)