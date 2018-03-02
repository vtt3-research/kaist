import torch

class EqSampler(object):


    def __init__(self, data_source):
        self.num_samples = len(data_source)
        pass

    def __iter__(self):
        return iter(torch.randperm(self.num_samples).long())

    def __len__(self):
        return self.num_samples