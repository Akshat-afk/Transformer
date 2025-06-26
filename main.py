import WordEmbedding
import PositionalEncoding
import torch
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

vocabulary = {'pen':0, 'ball':1}

# Try to change the vocabulary_size and number_of_embeddings
model = WordEmbedding.WordEmbeddingNN(vocabulary_size=len(vocabulary), number_of_embeddings=3).to('mps')
input_words = ['ball', 'pen', 'ball']
input_word_indices = torch.tensor([vocabulary[word] for word in input_words]).to(device)
print(f"Input words:{input_words}\nInput word indices:{input_word_indices}")
word_embeddings = model(input_word_indices)
#Printing the Word Embeddings
print("Word Embeddings:\n")
for word, vector in zip(input_words, word_embeddings.detach().cpu()):
    print(f"{word}: {vector.numpy()}")

positional_encodings = PositionalEncoding.positionalEncoding(len(input_words),3)
print(positional_encodings)

print(word_embeddings+positional_encodings)