import argparse
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append("./src")

import torch
from utils import use_torch_npu

npu_available = use_torch_npu()
if npu_available:
    print("NPU is available.")
else:
    print("NPU is not available.")

from utils import set_seed, ModelNameGenerator
from utils import count_learnable_parameters
from utils import LayerWiseFPCUDATimeProfiler
import models as models
from data_module import SCIFARDataModule


def parse_args():
    parser = argparse.ArgumentParser(description='Classify Sequential CIFAR')
    parser.add_argument('-neuron', "--neuron_type", default='LIF', type=str)
    parser.add_argument(
        "-sc",
        "--spike_compressor",
        default="IdentitySpikeCompressor",
        type=str
    )
    parser.add_argument(
        '-net', "--network", default="MESequentialCIFARNet", type=str
    )
    parser.add_argument(
        '--data_dir', type=str, default="/home/ma-user/work/datasets/CIFAR10"
    )
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("-nc", "--num_classes", default=10, type=int)
    parser.add_argument('-d', '--device', default='cuda:0', type=str)
    parser.add_argument("-ss", "--set_seed", type=int, default=2024)

    args = parser.parse_args()
    args.decay_lambda = 0.5
    args.channels = 128
    args.batch_size = 128
    args.num_workers = 4
    args.lomo = False
    args.amp = False
    args.warmup = 10
    return args


def main():
    args = parse_args()
    print(args)

    set_seed(args.set_seed)

    run_name = ModelNameGenerator(
        proj=f"sequential-cifar{args.num_classes}-me"
    ).generate(args)
    log_path = Path(args.log_dir) / f"SCIFAR{args.num_classes}"
    if not log_path.exists():
        log_path.mkdir(parents=True)
    log_path = log_path / (run_name+".time-prof.txt")

    data_module = SCIFARDataModule(
        args.data_dir, args.num_classes, args.batch_size, args.num_workers
    )
    data_module.setup(stage="fit")
    val_data_loader = data_module.val_dataloader()

    net = getattr(models, args.network)(
        channels=args.channels,
        neuron_type=args.neuron_type,
        spike_compressor=args.spike_compressor,
        num_classes=args.num_classes,
        decay_lambda=args.decay_lambda,
        T=32,  # for PSN
        k=8,  # for SlidingPSN
    )
    print(net)
    print("Number of learnable parameters: ", count_learnable_parameters(net))
    net = net.to(args.device)

    profiler = LayerWiseFPCUDATimeProfiler(
        (net.conv, net.fc, net.decode),
        search_mode=("direct_children", "self", "self"),
        model_names=("conv", "fc", "decode"),
        instances=(torch.nn.Module,),
        filename=log_path,
        warmup=args.warmup,
    )

    with torch.no_grad():
        for img, label in tqdm(val_data_loader):
            img, label = img.float().to(args.device), label.to(args.device)
            img = img.permute(3, 0, 1, 2)  # [W, N, C, H]; W acts as T
            y = net(img)

    profiler.clear_hooks()
    profiler.profile()


if __name__ == '__main__':
    main()
