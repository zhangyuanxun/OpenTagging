import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, context_output, attribute_output):
        seq_len = context_output.size()[1]
        attribute_output = attribute_output.unsqueeze(1).repeat(1, seq_len, 1)
        cos_sim = torch.cosine_similarity(context_output, attribute_output, -1)
        cos_sim = cos_sim.unsqueeze(-1)
        outputs = context_output * cos_sim
        return outputs
