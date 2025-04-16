import sys

sys.path.append("./src")

import torch
import torch.nn.functional as F

import models
from utils import count_learnable_parameters
from utils import set_seed


def test_sew18_imagenet():
    set_seed(2025)

    B = 2
    net = models.sew_resnet18("SJLIF", T=4, detach_reset=True)
    net = net.to("cuda")
    print(net)
    print("Number of learnable parameters: ", count_learnable_parameters(net))

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    x = torch.randn(2, 3, 224, 224).to("cuda")
    y = net(x)
    print(y.shape)
    assert y.shape == (B, 1000)

    with torch.no_grad():
        target = x.sum(dim=(1, 2, 3)) >= 0
        target = target.long()
    loss = F.cross_entropy(y, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


if __name__ == "__main__":
    test_sew18_imagenet()
