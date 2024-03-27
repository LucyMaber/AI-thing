import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, nhead, num_layers):
        super(AutoEncoder, self).__init__()

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_size,
                nhead=nhead,
                dim_feedforward=hidden_size,
            ),
            num_layers=num_layers,
        )

        # Transformer Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=input_size,
                nhead=nhead,
                dim_feedforward=hidden_size,
            ),
            num_layers=num_layers,
        )

    def forward(self, x):
        # Encoder forward pass
        encoded = self.encoder(x)

        # Decoder forward pass with memory argument
        decoded = self.decoder(encoded, memory=encoded)

        return decoded

# Example usage
input_size = 64  # Input size of the autoencoder
hidden_size = 128  # Hidden size of the transformer layers
nhead = 4  # Number of attention heads
num_layers = 3  # Number of transformer layers

# Create an instance of the AutoEncoder
autoencoder = AutoEncoder(input_size, hidden_size, nhead, num_layers)

# Generate a random input tensor
input_tensor = torch.randn(10, 32, input_size)  # (sequence length, batch size, input size)

# Forward pass through the autoencoder
output_tensor = autoencoder(input_tensor)
print(output_tensor.shape)  # (sequence length, batch size, input size)