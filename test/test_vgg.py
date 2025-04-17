import sys

sys.path.append("./src")

import torch
import torch.nn.functional as F

import models
from utils import count_learnable_parameters
from utils import set_seed

DEVICE = "cpu"


def make_parameters_equal(net, reference_net):
    """Assume that `net` and `reference_net` have the parameter order.
    """
    for p, ref_p in zip(net.parameters(), reference_net.parameters()):
        p.data = ref_p.data.clone()


def relative_distance(x, y):
    return torch.max(torch.abs(x - y) / torch.abs(x)).item()


def _test_vgg_cifar10dvs(net):
    set_seed(2025)
    B = 3
    print(net)
    print("Number of learnable parameters: ", count_learnable_parameters(net))

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    x = torch.randn(B, 10, 2, 48, 48).to(DEVICE)
    y = net(x)
    print(y.shape)
    assert y.shape == (B, 10)

    with torch.no_grad():
        target = x.sum(dim=(1, 2, 3, 4)) >= 0
        target = target.long()
    loss = F.cross_entropy(y, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def test_vgg_cifar10dvs():
    net = models.CIFAR10DVSVGG(T=10, neuron_type="SJLIF", detach_reset=True)
    net = net.to(DEVICE)
    _test_vgg_cifar10dvs(net)


def test_vgg_cifar10dvs_checkpointing():
    net = models.MECIFAR10DVSVGG(
        T=10,
        neuron_type="HandWrittenLIF",
        spike_compressor="BitSpikeCompressor",
        detach_reset=True
    )
    net = net.to(DEVICE)
    _test_vgg_cifar10dvs(net)


def test_vgg_equality_cifar10dvs():
    net1 = models.CIFAR10DVSVGG(
        T=10,
        neuron_type="HandWrittenLIF",
        dropout=0.,
        detach_reset=True,
    )
    net1 = net1.to(DEVICE)

    net2 = models.MECIFAR10DVSVGG(
        T=10,
        neuron_type="HandWrittenLIF",
        spike_compressor="BitSpikeCompressor",
        dropout=0.,
        detach_reset=True,
    )
    net2 = net2.to(DEVICE)
    make_parameters_equal(net2, net1)
    print(net2)

    B = 3
    x = torch.randn(B, 10, 2, 48, 48).to(DEVICE)
    y1 = net1(x)
    y2 = net2(x)
    print(y1, y2)
    assert torch.allclose(y1, y2)

    with torch.no_grad():
        target = x.sum(dim=(1, 2, 3, 4)) >= 0
        target = target.long()
    loss1 = F.cross_entropy(y1, target)
    loss2 = F.cross_entropy(y2, target)
    assert torch.allclose(loss1, loss2)

    loss1.backward()
    loss2.backward()

    for ((n1, p1), (n2, p2)) in zip(
        net1.named_parameters(),
        net2.named_parameters(),
    ):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5), (
            n1,
            n2,
            F.cosine_similarity(p1.grad.flatten(), p2.grad.flatten(), dim=0),
            F.l1_loss(p1.grad.flatten(), p2.grad.flatten()),
            relative_distance(p1.grad.flatten(), p2.grad.flatten()),
        )


if __name__ == "__main__":
    # test_vgg_cifar10dvs()
    # test_vgg_cifar10dvs_checkpointing()
    test_vgg_equality_cifar10dvs()
