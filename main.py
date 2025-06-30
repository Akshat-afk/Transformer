import torch
from Transformer import Transformer
import torch.nn as nn
import torch.optim as optim
import tensorflow_datasets as tfds
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Load a small text dataset (e.g., tiny Shakespeare)
ds = tfds.load('tiny_shakespeare', split='train', shuffle_files=True)
text_data = next(iter(tfds.as_numpy(ds)))['text'].decode('utf-8')

# Simple vocabulary creation
vocab = sorted(set(text_data))
vocabulary = {ch: idx for idx, ch in enumerate(vocab)}
reverse_vocab = {idx: ch for ch, idx in vocabulary.items()}

# Sample input sequence
input_text = text_data[:len(text_data) // 4]  # Use 1/4th of the dataset

# Prepare training batches
sequence_len = 64  # Define fixed sequence length
inputs = []
targets = []
for i in range(0, len(input_text) - sequence_len):
    seq = input_text[i:i+sequence_len]
    tgt = input_text[i+1:i+sequence_len+1]
    inputs.append([vocabulary[ch] for ch in seq])
    targets.append([vocabulary[ch] for ch in tgt])
input_indices = torch.tensor(inputs, dtype=torch.long).to(device)
target = torch.tensor(targets, dtype=torch.long).to(device)

# Create DataLoader for batching
from torch.utils.data import DataLoader, TensorDataset

batch_size = 32
dataset = TensorDataset(input_indices, target)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Parameters
vocabulary_size = len(vocabulary)
embedding_dim = 64
sequence_len = input_indices.size(1)

# Initialize Transformer
model = Transformer(
    vocabulary_size=vocabulary_size,
    number_of_embeddings=embedding_dim,
    sequence_len=sequence_len,
    input_dimensions=64,
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# for epoch in range(10):  # Reduced for efficiency
#     model.train()
#     epoch_loss = 0
#     for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
#         optimizer.zero_grad()
#         output = model(x)
#         loss = criterion(output.view(-1, vocabulary_size), y.view(-1))
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# torch.save(model.state_dict(), "transformer_trained.pth")

# Inference
model.eval()
with torch.no_grad():
    output = model(input_indices[:1])
    predicted_indices = output.argmax(dim=-1)
    predicted_tokens = [reverse_vocab[i] for i in predicted_indices[0].tolist()]
    print("Predicted Tokens:", predicted_tokens)