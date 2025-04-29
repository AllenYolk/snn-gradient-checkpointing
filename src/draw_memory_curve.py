import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "nature", "grid", "no-latex"])

MB = 1024 * 1024
W, H = plt.rcParams["figure.figsize"]
current_palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def get_curve(data_path):
    data_dict = torch.load(data_path)
    forward_start_memory = data_dict["forward_start_memory"]
    forward_end_memory = data_dict["forward_end_memory"]
    forward_peak_memory = data_dict["forward_peak_memory"]
    backward_start_memory = data_dict["backward_start_memory"]
    backward_end_memory = data_dict["backward_end_memory"]
    backward_peak_memory = data_dict["backward_peak_memory"]

    fp_curve = []
    bp_curve = []
    for k in forward_start_memory.keys():
        fp_curve.append(forward_start_memory[k] / MB)
        fp_curve.append(forward_peak_memory[k] / MB)
        fp_curve.append(forward_end_memory[k] / MB)
    for k in backward_start_memory.keys():
        bp_curve.append(backward_start_memory[k] / MB)
        bp_curve.append(backward_peak_memory[k] / MB)
        bp_curve.append(backward_end_memory[k] / MB)
    return fp_curve, bp_curve, max(max(fp_curve), max(bp_curve))


def plot_curve(ax, fp_curve, bp_curve, label, color):
    ax.plot(
        np.arange(len(fp_curve)),
        fp_curve,
        marker="o",
        markersize=2,
        label=label,
        color=color
    )
    ax.plot(
        np.arange(
            len(fp_curve),
            len(fp_curve) + len(bp_curve),
        ),
        bp_curve,
        marker="d",
        markersize=2,
        color=color
    )


parser = argparse.ArgumentParser()
parser.add_argument("--data_path_1", type=str)
parser.add_argument("--data_path_2", type=str, default=None)
parser.add_argument("--data_path_3", type=str, default=None)
args = parser.parse_args()

fp_curve1, bp_curve1, p1 = get_curve(args.data_path_1)
fp_curve2, bp_curve2, p2 = get_curve(args.data_path_2)
fp_curve3, bp_curve3, p3 = get_curve(args.data_path_3)

f, ax = plt.subplots(figsize=(W * 1.5, H))
plot_curve(ax, fp_curve1, bp_curve1, "BPTT", current_palette[0])
plot_curve(ax, fp_curve2, bp_curve2, "G.C.", current_palette[1])
plot_curve(ax, fp_curve3, bp_curve3, "G.C.+1bit", current_palette[2])
ax.axhline(p3, color="#FF6F61", linewidth=0.5, label="Peak Memory")
ax.set_ylabel("Memory Usage (MB)")
ax.set_xticklabels([])
ax.grid(linewidth=0.25, alpha=0.2)
ax.legend()
plt.tight_layout()
plt.savefig("./imgs/memory_curve.png", dpi=300)
plt.show()
