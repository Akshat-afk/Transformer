import torch
from Encoder import Encoder
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Define vocabulary and input
vocabulary = {'pen': 0, 'ball': 1}
input_words = ['ball', 'pen', 'ball']
input_indices = torch.tensor([vocabulary[word] for word in input_words]).unsqueeze(0).to(device)  # [1, seq_len]

# Parameters
vocabulary_size = len(vocabulary)
embedding_dim = 3
sequence_len = input_indices.size(1)

# Initialize Encoder
encoder = Encoder(
    vocabulary_size=vocabulary_size,
    number_of_embeddings=embedding_dim,
    sequence_len=sequence_len,
    input_dimensions=3
).to(device)

# Forward pass
output = encoder(input_indices)

# Output
print("Input words:", input_words)
print("\nEncoder Output:\n", output)