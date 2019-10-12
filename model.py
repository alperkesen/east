import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EAST(nn.Module):
    def __init__(self, extractor="vgg16"):
        super(EAST, self).__init__()

        if extractor == "vgg16":
            self.extractor = models.vgg16(pretrained=True)

            for param in self.extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        self.pool2 = self.extractor.features[0:10]
        self.pool3 = self.extractor.features[10:17]
        self.pool4 = self.extractor.features[17:24]
        self.pool5 = self.extractor.features[24:]

        f4 = self.pool2(x)
        f3 = self.pool3(f4)
        f2 = self.pool4(f3)
        f1 = self.pool5(f2)

        self.feature_maps = [f1, f2, f3, f4]
        
