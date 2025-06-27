import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

#3 networks
#Key, Value, Query

# #Key
# class KeyNN(nn.Module):
#     def __init__(self, input_dimensions, hidden_dimensions):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dimensions,hidden_dimensions),
#             nn.ReLU(),
#             nn.Linear(hidden_dimensions,input_dimensions)
#         )

#     def forward(self, x):
#         return self.network(x)
    
# #Query
# class QueryNN(nn.Module):
#     def __init__(self, input_dimensions, hidden_dimensions):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dimensions, hidden_dimensions),
#             nn.ReLU(),
#             nn.Linear(hidden_dimensions, input_dimensions)
#         )

#     def forward(self, x):
#         return self.network(x)

# #Value
# class ValueNN(nn.Module):
#     def __init__(self, input_dimensions, hidden_dimensions):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dimensions, hidden_dimensions),
#             nn.ReLU(),
#             nn.Linear(hidden_dimensions, input_dimensions)
#         )

#     def forward(self, x):
#         return self.network(x)
    
class SelfAttention(nn.Module):
    def __init__(self, input_dimensions):
        super().__init__()
        self.key_network = nn.Linear(input_dimensions, input_dimensions)
        self.query_network = nn.Linear(input_dimensions, input_dimensions)
        self.value_network = nn.Linear(input_dimensions, input_dimensions)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        Query = self.query_network(x)
        Value = self.value_network(x)
        Key = self.key_network(x)

        scaling = Query.size(-1) ** 0.5
        similarity_score = torch.matmul(Query, Key.transpose(-2,-1)) / scaling
        attention_probabilities = self.softmax(similarity_score)
        attention_output = torch.matmul(attention_probabilities, Value)

        return attention_output

def testing():
    model = SelfAttention(2).to(device)
    print(model)
