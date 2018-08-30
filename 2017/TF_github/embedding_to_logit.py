import torch.nn as nn

class embedding_to_logit(nn.Module):

    def __init__(self, emb_size = 128, num_classes=31):
        super(embedding_to_logit, self).__init__()
        self.fc8 = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.fc8(x)
        return x
