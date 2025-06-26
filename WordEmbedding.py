import torch.nn as nn
import torch

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
class WordEmbeddingNN(nn.Module):
    def __init__(self, vocabulary_size, number_of_embeddings):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(vocabulary_size, number_of_embeddings))
    
    def forward(self, x):
        return self.embedding[x]
    
# Testing
def testing():
    vocabulary = {'pen':0, 'ball':1}

    # Try to change the vocabulary_size and number_of_embeddings
    model = WordEmbeddingNN(vocabulary_size=len(vocabulary), number_of_embeddings=3).to('mps')

    input_words = ['ball', 'pen', 'ball']
    input_word_indices = torch.tensor([vocabulary[word] for word in input_words]).to(device)
    print(f"Input words:{input_words}\nInput word indices:{input_word_indices}")

    output = model(input_word_indices)
    print(output.type,output)
    #Printing the Word Embeddings
    print("Word Embeddings:\n")
    for word, vector in zip(input_words, output.detach().cpu()):
        print(f"{word}: {vector.numpy()}")
