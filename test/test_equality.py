import sys

sys.path.insert(0, "./src")

import torch

from models import VanillaLIF, HandWrittenLIF, MELIF


def test_forward_equality():
    T = 1000
    N = 10
    C = 100
    x = torch.randn(T, N, C) + 0.6

    # VanillaLIF
    s_vanilla = VanillaLIF()(x)

    # HandWrittenLIF
    s_handwritten = HandWrittenLIF()(x)

    s_me = MELIF()(x)

    assert torch.allclose(s_vanilla, s_handwritten, atol=1e-5)
    assert torch.allclose(s_vanilla, s_me, atol=1e-5)
    print(f"VanillaLIF spike rate: {s_vanilla.mean().item()}")
    print(f"HandWrittenLIF spike rate: {s_handwritten.mean().item()}")
    print(f"MELIF spike rate: {s_me.mean().item()}")


def test_backward_equality():
    T = 1000
    N = 10
    C = 100
    x = torch.randn(T, N, C) + 0.6
    grad_s = torch.randn(T, N, C)

    # VanillaLIF
    x1 = x.clone()
    x1.requires_grad = True
    s_vanilla = VanillaLIF()(x1)
    s_vanilla.backward(grad_s)

    # HandWrittenLIF
    x2 = x.clone()
    x2.requires_grad = True
    s_handwritten = HandWrittenLIF()(x2)
    s_handwritten.backward(grad_s)

    # MELIF
    x3 = x.clone()
    x3.requires_grad = True
    s_me = MELIF()(x3)
    s_me.backward(grad_s)

    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(x1.grad, x3.grad, atol=1e-5)
    print("Grad_x[t=0] for some VanillaLIF: ", x1.grad[0, 0, :10])
    print("Grad_x[t=0] for some HandWrittenLIF: ", x2.grad[0, 0, :10])
    print("Grad_x[t=0] for some MELIF: ", x3.grad[0, 0, :10])


if __name__ == "__main__":
    test_forward_equality()
    test_backward_equality()
