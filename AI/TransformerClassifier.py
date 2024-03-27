import torch
import torch.nn as nn
from transformers import AutoTokenizer

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout_prob):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer(embedded, embedded)
        # The transformer output is a sequence, but for classification, we typically use only the output of the first token
        cls_output = transformer_output.mean(dim=1)
        logits = self.fc(cls_output)
        return logits

    def set_output_dim(self, output_dim):
        # This function allows you to change the number of output classes dynamically
        self.fc = nn.Linear(self.fc.in_features, output_dim)

# # Example usage
# input_dim = 10000  # Size of the vocabulary
# hidden_dim = 128
# output_dim = 2  # Initial number of classes
# num_heads = 4
# num_layers = 2
# dropout_prob = 0.1

# model = TransformerClassifier(input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout_prob)

# # Dummy input tensor
# dummy_input = torch.randint(0, input_dim, (32, 20))  # Batch size of 32, sequence length of 20
# print(dummy_input)
# # Forward pass
# # output = model(dummy_input)
# # print(output.shape)

# # # Change the number of output classes dynamically
# # new_output_dim = 5
# # model.set_output_dim(new_output_dim)

# # # Forward pass with the updated number of output classes
# # output = model(dummy_input)
# # print(output.shape)
