import argparse
import torch
import numpy as np

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "nature", "grid", "no-latex"])

MB = 1024 * 1024
W, H = plt.rcParams["figure.figsize"]


def get_curve(data_path):
    data_dict = torch.load(data_path)
    forward_start_memory = data_dict["forward_start_memory"]
    forward_end_memory = data_dict["forward_end_memory"]
    forward_peak_memory = data_dict["forward_peak_memory"]
    backward_start_memory = data_dict["backward_start_memory"]
    backward_end_memory = data_dict["backward_end_memory"]
    backward_peak_memory = data_dict["backward_peak_memory"]

    curve = []
    for k in forward_start_memory.keys():
        print(k)
        curve.append(forward_start_memory[k] / MB)
        curve.append(forward_peak_memory[k] / MB)
        curve.append(forward_end_memory[k] / MB)
    for k in reversed(backward_start_memory.keys()):
        print(k)
        curve.append(backward_start_memory[k] / MB)
        curve.append(backward_peak_memory[k] / MB)
        curve.append(backward_end_memory[k] / MB)
    return curve


parser = argparse.ArgumentParser()
parser.add_argument("--data_path_1", type=str)
parser.add_argument("--data_path_2", type=str)
args = parser.parse_args()

curve1 = get_curve(args.data_path_1)
curve2 = get_curve(args.data_path_2)

f, ax = plt.subplots(figsize=(W * 1.5, H))
ax.plot(curve1, marker="o", markersize=2, label="Vanilla")
ax.plot(curve2, marker="o", markersize=2, label="G.C.")
ax.set_ylabel("Memory Usage (MB)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("./imgs/memory_curve.png", dpi=300)
plt.show()
