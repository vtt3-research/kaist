import torch
import torch.nn as nn
from DA_layer import autodial2d, batchnorm2d, autodial_grad2d

class DomainAlignment(nn.Module):
    def __init__(self, batch_size, split):
        super(DomainAlignment, self).__init__()
        self.batch_size = batch_size
        self.split = split
        self.eps = 1e-5

    def forward(self, input, alpha):
        if input.size(0) == self.batch_size:
            output = autodial2d(input, alpha, self.batch_size, self.split)
        else:
            output = batchnorm2d(input)
        return output
