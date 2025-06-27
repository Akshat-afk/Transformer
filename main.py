import WordEmbedding
import PositionalEncoding
import SelfAttention

import torch
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

vocabulary = {'pen':0, 'ball':1}
input_words = ['ball', 'pen', 'ball']

number_of_embeddings = 3
input_dimensions = len(input_words)

# Try to change the vocabulary_size and number_of_embeddings
model = WordEmbedding.WordEmbeddingNN(vocabulary_size=len(vocabulary), number_of_embeddings=number_of_embeddings).to('mps')


input_word_indices = torch.tensor([vocabulary[word] for word in input_words]).to(device)
print(f"Input words:{input_words}\nInput word indices:{input_word_indices}")
word_embeddings = model(input_word_indices)
#Printing the Word Embeddings
print("Word Embeddings:\n")
for word, vector in zip(input_words, word_embeddings.detach().cpu()):
    print(f"{word}: {vector.numpy()}")

positional_encodings = PositionalEncoding.positionalEncoding(len(input_words),3)

positional_encoded_words = word_embeddings+positional_encodings

Self_Attention = SelfAttention.SelfAttention(input_dimensions).to(device)

# Move positional_encoded_words to the same device as the model
positional_encoded_words = positional_encoded_words.to(device)

output = Self_Attention(positional_encoded_words)

print("\nSelf-Attention Output:\n")
print(output)