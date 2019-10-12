import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EAST(nn.Module):
    def __init__(self, extractor="vgg16"):
        super(EAST, self).__init__()

        if extractor == "vgg16":
            self.extractor = models.vgg16(pretrained=True)

        
