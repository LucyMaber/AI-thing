import torch
import torch.nn as nn
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
