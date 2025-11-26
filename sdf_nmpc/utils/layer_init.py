import torch
import numpy as np


def init_conv_layers(layer):
    """Recursive function to initialize all conv2d layers with xavier_uniform_ throughout the sub modules."""
    if type(layer) in [torch.nn.Conv2d, torch.nn.ConvTranspose2d]:
        torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('conv2d'))
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
    for ll in layer.children():
        init_conv_layers(ll)


def init_linear_layer_sine(layer, w0):
    """Recursive function to initialize a Linear layer (when used with Sine activation).
    Taken after code at https://ishit.github.io/modsine/ and https://github.com/vsitzmann/siren/blob/master/modules.py#L622
    """
    if type(layer) == torch.nn.Linear:
        num_input = layer.weight.size(-1)
        torch.nn.init.uniform_(layer.weight, -np.sqrt(6. / num_input) / w0, np.sqrt(6. / num_input) / w0)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)
    for ll in layer.children():
        init_linear_layer_sine(ll, w0)
