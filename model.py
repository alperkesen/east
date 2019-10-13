import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class EAST(nn.Module):
    def __init__(self, extractor="vgg16"):
        super(EAST, self).__init__()

        if extractor == "vgg16":
            self.extractor = models.vgg16(pretrained=True)

            for param in self.extractor.parameters():
                param.requires_grad = False

            self.pool2 = self.extractor.features[0:10]
            self.pool3 = self.extractor.features[10:17]
            self.pool4 = self.extractor.features[17:24]
            self.pool5 = self.extractor.features[24:]

        self.unpool1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.unpool2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.unpool3 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.conv1_1 = nn.Conv2d(1024, 256, 1)
        self.conv1_2 = nn.Conv2d(512, 128, 1)
        self.conv1_3 = nn.Conv2d(256, 64, 1)

        self.conv3_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_4 = nn.Conv2d(64, 32, 3, padding=1)

        self.unpools = [self.unpool1, self.unpool2, self.unpool3]
        self.convs1 = [self.conv1_1, self.conv1_2, self.conv1_3]
        self.convs3 = [self.conv3_1, self.conv3_2, self.conv3_3, self.conv3_4]

        self.conv1_4 = nn.Conv2d(32, 1, 1)
        self.conv1_5 = nn.Conv2d(32, 4, 1)
        self.conv1_6 = nn.Conv2d(32, 1, 1)
        self.conv1_7 = nn.Conv2d(32, 8, 1)

    def forward(self, x):
        # Feature extraction
        
        f4 = self.pool2(x)
        f3 = self.pool3(f4)
        f2 = self.pool4(f3)
        f1 = self.pool5(f2)

        f = [f1, f2, f3, f4]

        # Feature merging
        
        h = [None, None, None, None]
        g = [None, None, None, None]

        for i in range(4):
            if i == 0:
                h[i] = f[i]
            else:
                conv1, conv3 = self.convs1[i-1], self.convs3[i-1]
                h[i] = conv3(conv1(torch.cat((g[i-1], f[i]), dim=1)))

            if i <= 2:
                unpool = self.unpools[i]
                g[i] = unpool(h[i])
            else:
                conv3 = self.convs3[i]
                g[i] = conv3(h[i])

        self.h = h
        self.g = g

        # Output layer

        x = self.g[3]
        self.score_map = self.conv1_4(x)

        # RBOX

        self.rbox = self.conv1_5(x)
        self.rbox_angle = self.conv1_6(x)

        # QUAD

        self.quad = self.conv1_7(x)


t = torch.tensor(np.random.rand(1, 3, 224, 224), dtype=torch.float)
m = EAST()

    
