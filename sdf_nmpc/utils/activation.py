import torch


class Sine(torch.nn.Module):
    """Implementation of the sine activation function from [Sitzmann et al, 2020] -- see https://www.vincentsitzmann.com/siren/
    The default frequency is 30 -- see [Sitzmann et al, 2020] Sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion
    """
    def __init__(self, w0=30.):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
