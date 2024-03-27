import torch
import torch.nn as nn
from gggg import MambaBlock
# from vectorDB import lookup_document
from charformer_pytorch import GBST
from transformers.models.llama import LlamaModel

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Layer normalization after self-attention
        self.norm1 = nn.LayerNorm(d_model)
        # Position-wise feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        # Layer normalization after feedforward
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head self-attention
        attn_output, _ = self.self_attn(x, x, x)
        # Residual connection and layer normalization
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        # Position-wise feedforward network
        ff_output = self.feedforward(x)
        # Residual connection and layer normalization
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x



class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        # Stack multiple TransformerDecoderLayers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # Forward pass through all layers
        for layer in self.layers:
            x = layer(x)
        return x



class TransformerLayer(nn.Module):
    def __init__(self, input_size, head_count=8, hidden_size=512, dropout=0.1):
        super(TransformerLayer, self).__init__()

        self.multihead_attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=head_count)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=input_size)

        self.feedforward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        self.layer_norm2 = nn.LayerNorm(normalized_shape=input_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Multihead Attention
        attention_output, _ = self.multihead_attention(x, x, x)
        x = x + self.dropout(attention_output)
        x = self.layer_norm1(x)

        # Feedforward layer
        feedforward_output = self.feedforward(x)
        x = x + self.dropout(feedforward_output)
        x = self.layer_norm2(x)

        return x

class CrossStitch(nn.Module):
    def __init__(self, input_sizes):
        super(CrossStitch, self).__init__()

        # Flatten layers for all inputs
        self.flatten_layers = nn.ModuleList([nn.Flatten() for _ in input_sizes])

        # Identity matrix initialization for the cross-stitch parameters
        self.input_sizes = input_sizes
        t = sum(input_sizes)
        self.cross_stitch = nn.Parameter(torch.eye(t), requires_grad=True)

    def forward(self, *inputs):
        input_reshaped = [flatten(input_i) for flatten, input_i in zip(self.flatten_layers, inputs)]

        # Concatenate flattened inputs
        concatenated_input = torch.cat(input_reshaped, dim=1)

        # Apply cross-stitch operation
        output = torch.matmul(concatenated_input, self.cross_stitch)

        # Reshape back to the original shapes
        output_split = torch.split(output, split_size_or_sections=self.input_sizes, dim=1)
        outputs = [output_i.view(input_i.size()) for output_i, input_i in zip(output_split, inputs)]

        return tuple(outputs)

class MultiTaskTransformer(nn.Module):
    def __init__(self, input_sizes,layers=3,task=2):
        super(MultiTaskTransformer, self).__init__()
        self.layers = layers
        self.task = task
        self.crossStitch = [CrossStitch(input_sizes) for _ in range(layers)]
        self.networks = nn.ModuleList([nn.ModuleList([TransformerLayer(input_size=input_sizes[task_i]) for task_i in range(task)]) for _ in range(layers)])
        # self.networks = nn.ModuleList([nn.ModuleList([MambaBlock(batch_size,seq_len, d_model=input_sizes[task_i],state_size=state_size) for task_i in range(task)]) for _ in range(layers)])
    
    def forward(self, *inputs):
        for layer in range(self.layers):
            # cross stitch
            inputs = self.crossStitch[layer](*inputs)
            # transformer
            inputs = [self.networks[layer][task_i](inputs[task_i]) for task_i in range(self.task)]
        return inputs

class AutoEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, nhead, num_layers):
                # Transformer Decoder
                self.decoder = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(
                        d_model=input_size,
                        nhead=nhead,
                        dim_feedforward=hidden_size,
                    ),
                    num_layers=num_layers,
                )

    

class PirateNetIR(nn.Module):
    def __init__(self, num_tokens, input_size, max_block_size, downsample_factor, score_consensus_attn, layers, task, max_hops):
        super(PirateNetIR, self).__init__()
        input_sizes =  [input_size] * task
        self.tokenizer = GBST(num_tokens=num_tokens, dim=input_size, max_block_size=max_block_size, downsample_factor=downsample_factor, score_consensus_attn=score_consensus_attn)
        self.hops = max_hops
        self.module = MultiTaskTransformer(input_sizes, layers, task)
        self.fc_stop = nn.Linear(input_size, 1)
        self.task = task
        self.max_pool = nn.MaxPool1d(kernel_size=input_size)
        self.decoder = TransformerDecoder(input_size, input_size, 4, layers)
    
    def forward(self,inputs,mask,training_mode=False):
        inputs, mask = self.tokenizer(inputs, mask = mask)
        inputs = inputs.squeeze()
        inputs = ([inputs] * self.task)
        for _ in range(self.hops):
            inputs = self.module(*inputs)
            # autoencoder
            self.decoder(inputs[2])
            # stop condition
            stop = torch.sigmoid(self.fc_stop(inputs[0]).mean(dim=0))[0]
            print(stop)
            if (stop < 0.5) and not training_mode:  # Adjust the threshold if needed
                break
            # info Retrieval
            pooled_tensor, _ = torch.max(inputs[1], dim=0)
            print("pooled_tensor:",pooled_tensor.shape)
        return inputs
        

num_tokens = 257
input_size = 512
max_block_size = 4 
downsample_factor = 4
score_consensus_attn = True
task = 5
layers = 5
max_hops=5

pirate_net = PirateNetIR(num_tokens, input_size, max_block_size, downsample_factor, score_consensus_attn, layers, task, max_hops)
tokens = torch.randint(0, 257, (1, 2010)) # uneven number of tokens (1023)
mask   = torch.ones(1, 2010).bool()
tokens = pirate_net(tokens, mask = mask)
# print(tokens)
