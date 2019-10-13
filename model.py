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

        layer1 = nn.Sequential(nn.Conv2d(1024, 256, 1),
                               nn.BatchNorm2d(256),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(256, 256, 3, padding=1),
                               nn.BatchNorm2d(256),
                               nn.ReLU(inplace=True))

        layer2 = nn.Sequential(nn.Conv2d(512, 128, 1),
                               nn.BatchNorm2d(128),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(128, 128, 3, padding=1),
                               nn.BatchNorm2d(128),
                               nn.ReLU(inplace=True))

        layer3 = nn.Sequential(nn.Conv2d(256, 64, 1),
                               nn.BatchNorm2d(64),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(64, 64, 3, padding=1),
                               nn.BatchNorm2d(64),
                               nn.ReLU(inplace=True))

        layer4 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1),
                               nn.BatchNorm2d(32),
                               nn.ReLU(inplace=True))

        self.layers = [layer1, layer2, layer3, layer4]
        self.unpools = [nn.Upsample(scale_factor=2, mode="bilinear"),
                        nn.Upsample(scale_factor=2, mode="bilinear"),
                        nn.Upsample(scale_factor=2, mode="bilinear")]

        self.output_score = nn.Conv2d(32, 1, 1)
        self.output_textbox = nn.Conv2d(32, 4, 1)
        self.output_angle = nn.Conv2d(32, 1, 1)
        self.output_quad = nn.Conv2d(32, 8, 1)


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
            print(i)
            if i == 0:
                h[i] = f[i]
            else:
                h[i] = self.layers[i-1](torch.cat((g[i-1], f[i]), dim=1))

            if i <= 2:
                g[i] = self.unpools[i](h[i])
            else:
                g[i] = self.layers[i](h[i])

        self.h = h
        self.g = g

        # Output layer

        x = g[3]
        self.score_map = self.output_score(x)
        self.rbox_textbox = self.output_textbox(x)
        self.rbox_angle = self.output_angle(x)
        self.quad = self.output_quad(x)


t = torch.tensor(np.random.rand(1, 3, 224, 224), dtype=torch.float)
m = EAST()
