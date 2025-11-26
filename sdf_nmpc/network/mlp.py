import numpy as np
import torch


class Mlp(torch.nn.Module):
    """Simple MultiLayer Perceptron."""
    def __init__(self, size_in, size_out, layer_sizes, inner_act, out_act=torch.nn.Identity(), dropout_rate=0):
        super(Mlp, self).__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.layer_sizes = layer_sizes
        self.inner_act = inner_act
        self.out_act = out_act
        self.dropout = torch.nn.Dropout(dropout_rate)

        ## input layer
        self.layers = torch.nn.Sequential(torch.nn.Linear(size_in, layer_sizes[0]), self.inner_act, self.dropout)

        ## hiddem layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers.append(self.inner_act)
            self.layers.append(self.dropout)

        ## output layer
        self.layers.append(torch.nn.Linear(layer_sizes[-1], size_out))
        self.layers.append(self.out_act)

        # self.eval()

    def forward(self, x):
        return self.layers(x)
