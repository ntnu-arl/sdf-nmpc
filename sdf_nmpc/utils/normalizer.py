import torch


class Normalizer(torch.nn.Module):
    """
    Normalizes an input wrt the mean and std from the training data.
    The mean and std are registered as buffers and computed via compute_stats once, before training.
    """
    def __init__(self, dim=3, eps=1e-6):
        super(Normalizer, self).__init__()
        self.register_buffer('mean', torch.zeros(3))
        self.register_buffer('std', torch.ones(3))
        self.eps = eps  ## avoid division by zero

    def compute_stats(self, data):
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0, unbiased=False)

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)
