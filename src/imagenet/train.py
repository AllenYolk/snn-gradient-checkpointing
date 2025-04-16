import sys

sys.path.append("./src")

import torch

import models
from utils import count_learnable_parameters

net = models.sew_resnet18("SJLIF", T=4, detach_reset=True)
net = net.to("cuda")
print(net)
print("Number of learnable parameters: ", count_learnable_parameters(net))

x = torch.randn(2, 3, 224, 224).to("cuda")
y = net(x)
print(y.shape)
