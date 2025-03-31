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
    h1 = torch.arange(-30, 0, 0.005)
    h2 = torch.arange(0, 30, 0.005)
    H = torch.concat((h1, h2))  # make sure 0 is included
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

    max_sg_error = np.max(np.abs(Y - YQ))
    max_s_error = np.max(np.abs(S - SQ))
    argmax_sg_error = H[np.argmax(np.abs(Y - YQ))]
    argmax_s_error = H[np.argmax(np.abs(S - SQ))]

    # deal with border cases
    u_abs = torch.tensor(h_quantizer.clamp_abs, dtype=torch.float32)
    Y_u_abs = atan_derivative(u_abs)
    max_sg_error = max(max_sg_error, Y_u_abs)
    argmax_sg_error = np.inf if max_sg_error == Y_u_abs else argmax_sg_error

    if verbose:
        print("Max error of SG:", max_sg_error, "at H-V_th =", argmax_sg_error)
        print("Max error of S:", max_s_error, "at H-V_th =", argmax_s_error)

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

    x = torch.arange(-0.01, 0.01, 1e-7)
    h_quantizer = ClampProjHQuantizer(
        clamp_abs=5e2, dtype=torch.float8_e4m3fn, verbose=True
    )
    x_quantized = h_quantizer.quantize(x)
    x_dequantized = h_quantizer.dequantize(x_quantized)
    plt.plot(x.numpy(), x_dequantized.numpy(), label=r"float8$\_$e4m3fn")

    x = torch.arange(-0.01, 0.01, 1e-7)
    h_quantizer = ClampProjHQuantizer(
        clamp_abs=1e7, dtype=torch.float8_e5m2, verbose=True
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
    x = torch.arange(-1e-4, 1e-4, 1e-8)
    xx = x - MIN_POS_FLOAT[torch.float8_e5m2] / 2
    xx = xx.to(dtype=torch.float8_e5m2)
    xxx = xx.to(dtype=torch.float32)
    xxx = xxx + MIN_POS_FLOAT[torch.float8_e5m2] / 2

    zero = torch.tensor(0.0, dtype=torch.float32)
    zero = zero - MIN_POS_FLOAT[torch.float8_e5m2] / 2
    zero = zero.to(dtype=torch.float8_e5m2)
    zero = zero.to(dtype=torch.float32)
    zero = zero + MIN_POS_FLOAT[torch.float8_e5m2] / 2
    zero = zero.item()

    neg = torch.tensor(-1e-10, dtype=torch.float32)
    neg = neg - MIN_POS_FLOAT[torch.float8_e5m2] / 2
    neg = neg.to(dtype=torch.float8_e5m2)
    neg = neg.to(dtype=torch.float32)
    neg = neg + MIN_POS_FLOAT[torch.float8_e5m2] / 2
    neg = neg.item()

    plt.plot(x.numpy(), x.numpy(), label="Original")
    plt.plot(x.numpy(), xxx.numpy(), label="float8_e4m3fn")
    plt.title(f"f(0)={zero}, f(-1e-10)={neg}")
    plt.savefig("./imgs/float8_e5m2.png", dpi=600)
    plt.show()


def test_numerics():
    x_seq = torch.arange(-1, 1, 1e-7)
    clamp_abs = dtype_abs = 500
    y_seq = (x_seq+clamp_abs) / (2*clamp_abs)
    y_seq = y_seq*2*dtype_abs - dtype_abs

    z_seq = x_seq * dtype_abs / clamp_abs

    assert torch.allclose(y_seq, z_seq), "Numerics error!"


def search_best_clamp_range():
    u_abs_array = np.arange(1., 30, 0.001).astype(np.float32)

    print("For torch.float8_e4m3fn:")
    res1 = []
    min_error = np.inf
    best_u_abs = None
    for u_abs in u_abs_array:
        h_quantizer = ClampProjHQuantizer(
            clamp_abs=u_abs,
            dtype=torch.float8_e4m3fn,
        )
        max_sg_error, max_s_error = _test_quantization_error(h_quantizer)
        res1.append(max_sg_error)
        if max_sg_error < min_error:
            min_error = max_sg_error
            best_u_abs = u_abs
    print("\tBest u_abs:", best_u_abs, "Min error:", min_error)

    print("For torch.float8_e5m2:")
    res2 = []
    min_error = np.inf
    best_u_abs = None
    for u_abs in u_abs_array:
        h_quantizer = ClampProjHQuantizer(
            clamp_abs=u_abs,
            dtype=torch.float8_e5m2,
        )
        max_sg_error, max_s_error = _test_quantization_error(h_quantizer)
        res2.append(max_sg_error)
        if max_sg_error < min_error:
            min_error = max_sg_error
            best_u_abs = u_abs
    print("\tBest u_abs:", best_u_abs, "Min error:", min_error)

    plt.figure(
        figsize=(
            plt.rcParams["figure.figsize"][0] * 1.6,
            plt.rcParams["figure.figsize"][1]
        )
    )
    plt.plot(u_abs_array, res1, label="float8$\_$e4m3fn", linewidth=1)
    plt.plot(u_abs_array, res2, label="float8$\_$e5m2", linewidth=1)
    plt.xlabel(r"$u_{\mathrm{abs}}$")
    plt.ylabel("SG max abs error")
    plt.legend()
    plt.savefig("./imgs/h_quantization_error.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    #test_numerics()
    # print_atan_gradient_values()
    # test_quantization_error()
    # test_zero_neighborhood()
    #test_rounding()
    # test_casting()
    search_best_clamp_range()
