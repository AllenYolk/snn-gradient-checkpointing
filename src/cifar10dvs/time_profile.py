import sys

sys.path.append("./src")
sys.path.append("./src/cifar10dvs")

import torch
import torch.nn as nn
from tqdm import tqdm

import models
from utils import TMeanCrossEntropyLoss, profiler
from modules.checkpointing import memory_optimization
from data_module import CIFAR10DVSDataModule

DEVICE = "cuda:0"
LEVEL = 4
COMPRESS_X = True

if __name__ == "__main__":
    net = models.CIFAR10DVSVGG(
        10,
        neuron_type="MELIF",
        dropout=0.25,
        decay_lambda=0.5,
        k=2,
    )
    net = memory_optimization(
        net,
        (models.VGGBlock,),
        dummy_input=torch.zeros(32, 10, 2, 48, 48) + 0.9,
        compress_x=COMPRESS_X,
        level=LEVEL,
        verbose=True,
    ).to(DEVICE)

    loss_fn = TMeanCrossEntropyLoss()

    dm = CIFAR10DVSDataModule(
        "/export/home/data_allenyolk/CIFAR10DVS", T=10, batch_size=32, num_workers=4
    )
    dm.setup("fit")
    loader = dm.train_dataloader()

    target_model = (net.features, net.classifier)

    prof_fp = profiler.LayerWiseFPCUDATimeProfiler(
        target_model,
        model_names=("features", "classifier"),
        search_mode=("direct_children", "self"),
        instances=(nn.Module,),
    )
    prof_bp = profiler.LayerWiseBPCUDATimeProfiler(
        target_model,
        model_names=("features", "classifier"),
        search_mode=("direct_children", "self"),
        instances=(nn.Module,),
    )

    net.train()
    for x, y in tqdm(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = x.clone().detach()
        x.requires_grad = True
        out = net(x)
        loss = loss_fn(out, y)
        loss.backward()
        net.zero_grad()

    results_fp = prof_fp.export(output=False)
    prof_fp.close()
    results_bp = prof_bp.export(output=False)
    prof_bp.close()

    print(net)
    print("=" * 50, "FP", "=" * 50)
    print(results_fp)
    print("=" * 50, "BP", "=" * 50)
    print(results_bp)
