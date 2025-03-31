import sys

sys.path.insert(0, "./src")

import torch
from spikingjelly.activation_based import surrogate
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from models.neuron import atan_derivative
from models.compress.h_quantizer import *

plt.style.use(["science", "nature", "grid", "no-latex"])


def print_atan_gradient_values():
    x = torch.arange(-20, 20, 0.5)
    y = atan_derivative(x)
    for i, (xx, yy) in enumerate(zip(x, y)):
        print(i, xx.item(), yy.item())


def _test_quantization_error(h_quantizer, verbose=False):
    h1 = torch.arange(-11, 0, 0.001)
    h2 = torch.arange(0, 11, 0.001)
    H = torch.concat((h1, h2))
    HQ = h_quantizer.quantize(H)
    HQ = h_quantizer.dequantize(HQ)
    Y = atan_derivative(H)
    YQ = atan_derivative(HQ)
    S = (H >= 0.).float()
    SQ = (HQ >= 0.).float()
    H, Y, S = H.numpy(), Y.numpy(), S.numpy()
    HQ, YQ, SQ = HQ.numpy(), YQ.numpy(), SQ.numpy()

    if verbose:
        f, axes = plt.subplots(
            2,
            1,
            figsize=(
                plt.rcParams["figure.figsize"][0],
                plt.rcParams["figure.figsize"][1] * 2
            )
        )
        ax = axes[0]
        ax.plot(H, Y, label="Original")
        ax.plot(H, YQ, label="Quantized")
        ax.legend()
        ax.set_xlabel(r"H - $V_{th}$")
        ax.set_ylabel("surrogate derivative")
        ax = axes[1]
        ax.plot(H, S, label="Original")
        ax.plot(H, SQ, label="Quantized")
        ax.legend()
        ax.set_xlabel(r"H - $V_{th}$")
        ax.set_ylabel("spike")
        plt.savefig("./imgs/h_quantization.png", dpi=600)
        plt.show()

    max_sg_error = H[np.argmax(np.abs(Y - YQ))]
    max_s_error = H[np.argmax(np.abs(S - SQ))]
    if verbose:
        print(
            "Max error of SG:", np.max(np.abs(Y - YQ)), "at H-V_th =",
            max_sg_error
        )
        print(
            "Max error of S:", np.max(np.abs(S - SQ)), "at H-V_th =",
            max_s_error
        )

    return max_sg_error, max_s_error


def test_zero_neighborhood():
    h_quantizer = ClampProjHQuantizer(
        clamp_abs=1.,
        dtype=torch.float8_e4m3fn,
    )

    H = torch.tensor(-1e-7)
    QH = h_quantizer.quantize(H)
    HH = h_quantizer.dequantize(QH)
    print(H, QH, HH)

    SH = (H >= 0.).float()
    SHH = (HH >= 0.).float()
    assert torch.allclose(SH, SHH), "Zero neighborhood quantization error!"


def test_rounding():
    x = torch.arange(-0.01, 0.01, 1e-7)
    plt.plot(x.numpy(), x.numpy(), label="Original")

    h_quantizer = ClampProjHQuantizer(
        clamp_abs=5e2,
        dtype=torch.float8_e4m3fn,
    )
    x_quantized = h_quantizer.quantize(x)
    x_dequantized = h_quantizer.dequantize(x_quantized)
    plt.plot(x.numpy(), x_dequantized.numpy(), label=r"float8$\_$e4m3fn")

    h_quantizer = ClampProjHQuantizer(
        clamp_abs=5e3,
        dtype=torch.float8_e5m2,
    )
    x_quantized = h_quantizer.quantize(x)
    x_dequantized = h_quantizer.dequantize(x_quantized)
    plt.plot(x.numpy(), x_dequantized.numpy(), label=r"float8$\_$e5m2")

    plt.legend()
    plt.xlabel("$u$")
    plt.ylabel(r"$\hat{u}$")
    plt.title("Rounding Error")
    plt.savefig("./imgs/rounding_error_float.png", dpi=600)
    plt.show()


def test_casting():
    x = torch.arange(-0.03, 0.03, 1e-5)
    xx = x - MIN_POS_FLOAT[torch.float8_e4m3fn] / 2
    xx = xx.to(dtype=torch.float8_e4m3fn)
    xxx = xx.to(dtype=torch.float32)
    xxx = xxx + MIN_POS_FLOAT[torch.float8_e4m3fn] / 2

    zero = torch.tensor(0.0, dtype=torch.float32)
    zero = zero - MIN_POS_FLOAT[torch.float8_e4m3fn] / 2
    zero = zero.to(dtype=torch.float8_e4m3fn)
    zero = zero.to(dtype=torch.float32)
    zero = zero + MIN_POS_FLOAT[torch.float8_e4m3fn] / 2
    zero = zero.item()

    neg = torch.tensor(-1e-10, dtype=torch.float32)
    neg = neg - MIN_POS_FLOAT[torch.float8_e4m3fn] / 2
    neg = neg.to(dtype=torch.float8_e4m3fn)
    neg = neg.to(dtype=torch.float32)
    neg = neg + MIN_POS_FLOAT[torch.float8_e4m3fn] / 2
    neg = neg.item()

    plt.plot(x.numpy(), x.numpy(), label="Original")
    plt.plot(x.numpy(), xxx.numpy(), label="float8_e4m3fn")
    plt.title(f"f(0)={zero}, f(-1e-10)={neg}")
    plt.savefig("./imgs/float8_e4m3fn.png", dpi=600)
    plt.show()


def search_best_clamp_range():
    print("For torch.float8_e4m3fn:")

    print("For torch.float8_e5m2:")


if __name__ == "__main__":
    # print_atan_gradient_values()
    # test_quantization_error()
    test_zero_neighborhood()
    test_rounding()
    test_casting()
    #search_best_clamp_range()
