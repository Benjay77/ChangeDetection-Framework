import torch.nn as nn
from torchvision import models


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        # features = vgg16('D', True, True, 3).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.to_relu_5_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        for x in range(23, 30):
            self.to_relu_5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)  # [64, 1024, 1024]
        # print('h:', h.shape)
        h_relu_1_2 = h  # [64, 1024, 1024]
        # print('h_relu_1_2: ', h_relu_1_2.shape)
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h  # [128, 512, 512]
        # print('h_relu_2_2: ', h_relu_2_2.shape)
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h  # [256, 256, 256]
        # print('h_relu_3_3: ', h_relu_3_3.shape)
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h  # [512, 128, 128]
        # print('h_relu_4_3: ', h_relu_4_3.shape)
        h = self.to_relu_5_3(h)
        h_relu_5_3 = h  # [512, 64, 64]
        # print('h_relu_5_3: ', h_relu_5_3.shape)
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3, h_relu_5_3)
        return out
