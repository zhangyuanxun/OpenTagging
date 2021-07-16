import torch
import torch.nn as nn
import torch.nn.functional as F


class CRF(nn.Module):
    def __init__(self, tag_list, device):
        super(CRF, self).__init__()
        self.START_TAG = "[CLS]"
        self.STOP_TAG = "[SEP]"

        self.tag_map = tag_map
        self.num_tags = len(tag_list)
        self.device = device
        self.transitions = torch.randn(self.num_tags, self.num_tags)
