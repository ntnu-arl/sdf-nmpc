import numpy as np
import torch


class ResBlock(torch.nn.Module):
    def __init__(self, size_in, stride, bottleneck=False, use_batchnorm=False, dropout_rate=0):
        """Residual block implementation.
        size_in         -- number of features of the input volume
        stride          -- stride for spatial dimension reduction
        bottleneck      -- use a bottleneck block (1x1, 3x3, 1x1 convolutions) or standard residual block (3x3, 3x3 convolutions)
        use_batchnorm   -- use batchnorm or not
        dropout_rate    -- rate for dropout after terminal activation
        """
        super(ResBlock, self).__init__()
        bottleneck_dim_reduction = 4  # hardcoded since I've never seen anyone use some other number
        size_inner = size_in // bottleneck_dim_reduction
        size_out = size_in * stride
        activation = torch.nn.ReLU

        if bottleneck:
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(size_in, size_inner, kernel_size=1, padding=0, stride=stride, bias=not use_batchnorm),
                torch.nn.BatchNorm2d(size_inner) if use_batchnorm else torch.nn.Identity(),
                activation(),
                torch.nn.Conv2d(size_inner, size_inner, kernel_size=3, padding=1, stride=1, bias=not use_batchnorm),
                torch.nn.BatchNorm2d(size_inner) if use_batchnorm else torch.nn.Identity(),
                activation(),
                torch.nn.Conv2d(size_inner, size_out, kernel_size=1, padding=0, stride=1, bias=not use_batchnorm),
                torch.nn.BatchNorm2d(size_out) if use_batchnorm else torch.nn.Identity(),
            )
        else:
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(size_in, size_out, kernel_size=3, padding=1, stride=stride, bias=not use_batchnorm),
                torch.nn.BatchNorm2d(size_out) if use_batchnorm else torch.nn.Identity(),
                activation(),
                torch.nn.Conv2d(size_out, size_out, kernel_size=3, padding=1, stride=1, bias=not use_batchnorm),
                torch.nn.BatchNorm2d(size_out) if use_batchnorm else torch.nn.Identity(),
            )

        if stride == 1:
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(size_in, size_out, kernel_size=1, padding=0, stride=stride, bias=not use_batchnorm),
                torch.nn.BatchNorm2d(size_out) if use_batchnorm else torch.nn.Identity(),
            )

        self.term_activation = activation()
        self.term_dropout = torch.nn.Dropout2d(dropout_rate) if dropout_rate else torch.nn.Identity()

    def forward(self, input):
        output = self.layers(input)
        output += self.shortcut(input)
        output = self.term_activation(output)
        output = self.term_dropout(output)
        return output


class ResBlockDeconv(torch.nn.Module):
    """Residual deconvolution block implementation.
    size_in         -- number of features of the input volume
    stride          -- stride for spatial dimension reduction
    output_padding  -- output shape regularization
    bottleneck      -- use a bottleneck block (1x1, 3x3, 1x1 convolutions) or standard residual block (3x3, 3x3 convolutions)
    use_batchnorm   -- use batchnorm or not
    dropout_rate    -- rate for dropout after terminal activation
    """
    def __init__(self, size_in, stride, output_padding=0, bottleneck=False, use_batchnorm=False, dropout_rate=0):
        super(ResBlockDeconv, self).__init__()
        bottleneck_dim_reduction = 4
        size_inner = size_in // bottleneck_dim_reduction
        size_out = size_in // stride
        activation = torch.nn.ReLU

        if bottleneck:
            self.layers = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(size_in, size_inner, kernel_size=1, padding=0, stride=stride, output_padding=output_padding, bias=not use_batchnorm),
                torch.nn.BatchNorm2d(size_inner) if use_batchnorm else torch.nn.Identity(),
                activation(),
                torch.nn.ConvTranspose2d(size_inner, size_inner, kernel_size=3, padding=1, stride=1, output_padding=0, bias=not use_batchnorm),
                torch.nn.BatchNorm2d(size_inner) if use_batchnorm else torch.nn.Identity(),
                activation(),
                torch.nn.ConvTranspose2d(size_inner, size_out, kernel_size=1, padding=0, stride=1, output_padding=0, bias=not use_batchnorm),
                torch.nn.BatchNorm2d(size_out) if use_batchnorm else torch.nn.Identity(),
            )
        else:
            self.layers = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(size_in, size_out, kernel_size=3, padding=1, stride=stride, output_padding=output_padding, bias=not use_batchnorm),
                torch.nn.BatchNorm2d(size_out) if use_batchnorm else torch.nn.Identity(),
                activation(),
                torch.nn.ConvTranspose2d(size_out, size_out, kernel_size=3, padding=1, stride=1, output_padding=0, bias=not use_batchnorm),
                torch.nn.BatchNorm2d(size_out) if use_batchnorm else torch.nn.Identity(),
            )

        if stride == 1:
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(size_in, size_out, kernel_size=1, padding=0, stride=stride, output_padding=output_padding, bias=not use_batchnorm),
                torch.nn.BatchNorm2d(size_out),
            )

        self.term_activation = activation()
        self.term_dropout = torch.nn.Dropout2d(dropout_rate) if dropout_rate else torch.nn.Identity()

    def forward(self, input):
        output = self.layers(input)
        output += self.shortcut(input)
        output = self.term_activation(output)
        output = self.term_dropout(output)
        return output
