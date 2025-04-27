import sys

sys.path.insert(0, "./src")

import torch

from modules import SJLIF, HandWrittenLIF, MELIF


def test_forward_equality_not_detached():
    T = 512
    N = 10
    C = 100
    x = torch.randn(T, N, C) + 0.6
    x = x.to("cuda")

    sjlif = SJLIF(detach_reset=False, backend="torch").to("cuda")
    s_sj = sjlif(x)

    handwritten_lif = HandWrittenLIF(detach_reset=False).to("cuda")
    s_handwritten = handwritten_lif(x)

    melif = MELIF(detach_reset=False).to("cuda")
    s_me = melif(x)
    assert torch.allclose(s_sj, s_handwritten, atol=1e-5)
    assert torch.allclose(s_sj, s_me, atol=1e-5)


def test_backward_equality_not_detached():
    T = 512
    N = 10
    C = 100
    x = torch.randn(T, N, C) + 0.6
    grad_s = torch.randn(T, N, C)
    x = x.to("cuda")
    grad_s = grad_s.to("cuda")

    # VanillaLIF
    x1 = x.clone()
    x1.requires_grad = True
    sjlif = SJLIF(detach_reset=False, backend="torch").to("cuda")
    s_vanilla = sjlif(x1)
    s_vanilla.backward(grad_s)

    # HandWrittenLIF
    x2 = x.clone()
    x2.requires_grad = True
    handwritten_lif = HandWrittenLIF(detach_reset=False).to("cuda")
    s_handwritten = handwritten_lif(x2)
    s_handwritten.backward(grad_s)

    # HandWrittenLIF
    x3 = x.clone()
    x3.requires_grad = True
    melif = MELIF(detach_reset=False).to("cuda")
    s_me = melif(x3)
    s_me.backward(grad_s)

    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(x1.grad, x3.grad, atol=1e-5)

    print("Firing rate:", s_vanilla.mean().item())


def test_forward_equality_detached():
    T = 512
    N = 10
    C = 100
    x = torch.randn(T, N, C) + 0.6
    x = x.to("cuda")

    sjlif = SJLIF(
        decay_lambda=0.99,
        detach_reset=True,
        backend="torch",
    ).to("cuda")
    s_vanilla = sjlif(x)

    handwritten_lif = HandWrittenLIF(
        decay_lambda=0.99,
        detach_reset=True,
    ).to("cuda")
    s_handwritten = handwritten_lif(x)

    melif = MELIF(
        decay_lambda=0.99,
        detach_reset=True,
    ).to("cuda")
    s_me = melif(x)

    assert torch.allclose(s_vanilla, s_handwritten, atol=1e-5)
    assert torch.allclose(s_vanilla, s_me, atol=1e-5)
    print("Firing rate:", s_vanilla.mean().item())


def test_backward_equality_detached():
    T = 512
    N = 10
    C = 100
    x = torch.randn(T, N, C) + 0.6
    grad_s = torch.randn(T, N, C)
    x = x.to("cuda")
    grad_s = grad_s.to("cuda")

    # VanillaLIF
    x1 = x.clone()
    x1.requires_grad = True
    sjlif = SJLIF(
        decay_lambda=0.01,
        detach_reset=True,
        backend="torch",
    ).to("cuda")
    s_vanilla = sjlif(x1)
    s_vanilla.backward(grad_s)

    # HandWrittenLIF
    x2 = x.clone()
    x2.requires_grad = True
    handwritten_lif = HandWrittenLIF(
        decay_lambda=0.01,
        detach_reset=True,
    ).to("cuda")
    s_handwritten = handwritten_lif(x2)
    s_handwritten.backward(grad_s)

    # HandWrittenLIF
    x3 = x.clone()
    x3.requires_grad = True
    melif = MELIF(
        decay_lambda=0.01,
        detach_reset=True,
    ).to("cuda")
    s_me = melif(x3)
    s_me.backward(grad_s)

    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(x1.grad, x3.grad, atol=1e-5)
    print("Firing rate:", s_vanilla.mean().item())


if __name__ == "__main__":
    test_forward_equality_not_detached()
    test_backward_equality_not_detached()
    test_forward_equality_detached()
    test_backward_equality_detached()
