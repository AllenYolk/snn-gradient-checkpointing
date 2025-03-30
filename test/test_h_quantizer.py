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


def test_quantization_error():
    h_quantizer = ClampIntHQuantizer(
        clamp_range=(-10, 10),
        dtype=torch.uint8,
    )

    h1 = torch.arange(-20, 0, 0.001)
    h2 = torch.arange(0, 20, 0.001)
    H = torch.concat((h1, h2))
    HQ = h_quantizer.quantize(H)
    HQ = h_quantizer.dequantize(HQ)
    Y = atan_derivative(H)
    YQ = atan_derivative(HQ)
    S = (H >= 0.).float()
    SQ = (HQ >= 0.).float()
    H, Y, S = H.numpy(), Y.numpy(), S.numpy()
    HQ, YQ, SQ = HQ.numpy(), YQ.numpy(), SQ.numpy()

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
    ax.set_xlabel("H - V_\{th\}")
    ax.set_ylabel("surrogate derivative")
    ax = axes[1]
    ax.plot(H, S, label="Original")
    ax.plot(H, SQ, label="Quantized")
    ax.legend()
    ax.set_xlabel("H - V_\{th\}")
    ax.set_ylabel("spike")
    plt.show()

    print(
        "Max error of SG:", np.max(np.abs(Y - YQ)), "at H-V_th =",
        H[np.argmax(np.abs(Y - YQ))]
    )
    print(
        "Max error of S:", np.max(np.abs(S - SQ)), "at H-V_th =",
        H[np.argmax(np.abs(S - SQ))]
    )


def test_zero_point():
    h_quantizer = ClampIntHQuantizer(
        clamp_range=(-2.5, 2.5),
        dtype=torch.uint8,
    )

    H = torch.tensor(0.0)
    QH = h_quantizer.quantize(H)
    HH = h_quantizer.dequantize(QH)
    print("H=", H.item(), ", QH=", QH.item(), ", HH=", HH.item())
    assert H.item() == HH.item(), "Zero point quantization error!"


if __name__ == "__main__":
    #print_atan_gradient_values()
    test_quantization_error()
    test_zero_point()
