import sys

sys.path.append("./src")
sys.path.append("./src/imagenet/transformer")

import torch
import torch.nn.functional as F

import models
from utils import count_learnable_parameters
from utils import set_seed

DEVICE = "cuda"


def make_parameters_equal(net, reference_net):
    """Assume that `net` and `reference_net` have the parameter order.
    """
    for p, ref_p in zip(net.parameters(), reference_net.parameters()):
        p.data = ref_p.data.clone()


def relative_distance(x, y):
    return torch.max(torch.abs(x - y) / torch.abs(x)).item()


def _test_qkformer_imagenet(net):
    set_seed(2025)

    B = 2
    print(net)
    print("Number of learnable parameters: ", count_learnable_parameters(net))

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    x = torch.randn(B, 3, 224, 224).to(DEVICE)
    y = net(x)
    print(y.shape)

    with torch.no_grad():
        target = x.sum(dim=(1, 2, 3)) >= 0
        target = target.long()
    loss = F.cross_entropy(y, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def test_qkformer_imagenet():
    net = models.QKFormer(
        "SJLIF",
        T=4,
        embed_dims=192,
        num_heads=8,
        mlp_ratios=3,
        depths=8,
        detach_reset=True
    )
    net = net.to(DEVICE)
    _test_qkformer_imagenet(net)


def test_meqkformer_imagenet():
    net = models.FGCQKFormer(
        "HandWrittenLIF",
        "NullSpikeCompressor",
        T=4,
        embed_dims=108,
        num_heads=9,
        mlp_ratios=2,
        depths=10,
        detach_reset=True
    )
    net = net.to(DEVICE)
    _test_qkformer_imagenet(net)


def test_qkformer_equality_imagenet():
    net1 = models.QKFormer(
        "SJLIF",
        T=4,
        embed_dims=256,
        num_heads=8,
        mlp_ratios=4,
        depths=10,
        detach_reset=True
    )
    net1 = net1.to(DEVICE)

    net2 = models.FGCQKFormer(
        "HandWrittenLIF",
        "BitSpikeCompressor",
        T=4,
        embed_dims=256,
        num_heads=8,
        mlp_ratios=4,
        depths=10,
        detach_reset=True
    )
    net2 = net2.to(DEVICE)
    make_parameters_equal(net2, net1)
    print(net2)
    print("Number of learnable parameters: ", count_learnable_parameters(net2))

    B = 2
    x = torch.randn(B, 3, 224, 224).to(DEVICE)
    y1 = net1(x)
    y2 = net2(x)
    print(y1, y2)
    assert torch.allclose(y1, y2)

    with torch.no_grad():
        target = x.sum(dim=(1, 2, 3)) >= 0
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
    #test_qkformer_imagenet()
    #test_meqkformer_imagenet()
    test_qkformer_equality_imagenet()
