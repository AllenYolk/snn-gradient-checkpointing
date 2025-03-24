import sys

sys.path.insert(0, "./src")

import torch

from models import SJLIF, HandWrittenLIF, MELIF


def test_forward_equality_not_detached():
    T = 1000
    N = 10
    C = 100
    x = torch.randn(T, N, C) + 0.6

    s_sj = SJLIF(detach_reset=False, backend="torch")(x)
    s_handwritten = HandWrittenLIF(detach_reset=False)(x)
    s_me = MELIF(detach_reset=False)(x)

    assert torch.allclose(s_sj, s_handwritten, atol=1e-5)
    assert torch.allclose(s_sj, s_me, atol=1e-5)


def test_backward_equality_not_detached():
    T = 1000
    N = 10
    C = 100
    x = torch.randn(T, N, C) + 0.6
    grad_s = torch.randn(T, N, C)

    # VanillaLIF
    x1 = x.clone()
    x1.requires_grad = True
    s_vanilla = SJLIF(detach_reset=False, backend="torch")(x1)
    s_vanilla.backward(grad_s)

    # HandWrittenLIF
    x2 = x.clone()
    x2.requires_grad = True
    s_handwritten = HandWrittenLIF(detach_reset=False)(x2)
    s_handwritten.backward(grad_s)

    # HandWrittenLIF
    x3 = x.clone()
    x3.requires_grad = True
    s_me = MELIF(detach_reset=False)(x3)
    s_me.backward(grad_s)

    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(x1.grad, x3.grad, atol=1e-5)


def test_forward_equality_detached():
    T = 1000
    N = 10
    C = 100
    x = torch.randn(T, N, C) + 0.6

    s_vanilla = SJLIF(decay_lambda=0.99, detach_reset=True, backend="torch")(x)
    s_handwritten = HandWrittenLIF(decay_lambda=0.99, detach_reset=True)(x)
    s_me = MELIF(decay_lambda=0.99, detach_reset=True)(x)

    assert torch.allclose(s_vanilla, s_handwritten, atol=1e-5)
    assert torch.allclose(s_vanilla, s_me, atol=1e-5)


def test_backward_equality_detached():
    T = 1000
    N = 10
    C = 100
    x = torch.randn(T, N, C) + 0.6
    grad_s = torch.randn(T, N, C)

    # VanillaLIF
    x1 = x.clone()
    x1.requires_grad = True
    s_vanilla = SJLIF(decay_lambda=0.01, detach_reset=True, backend="torch")(x1)
    s_vanilla.backward(grad_s)

    # HandWrittenLIF
    x2 = x.clone()
    x2.requires_grad = True
    s_handwritten = HandWrittenLIF(decay_lambda=0.01, detach_reset=True)(x2)
    s_handwritten.backward(grad_s)

    # HandWrittenLIF
    x3 = x.clone()
    x3.requires_grad = True
    s_me = MELIF(decay_lambda=0.01, detach_reset=True)(x3)
    s_me.backward(grad_s)

    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(x1.grad, x3.grad, atol=1e-5)


if __name__ == "__main__":
    test_forward_equality_not_detached()
    test_backward_equality_not_detached()
    test_forward_equality_detached()
    test_backward_equality_detached()
