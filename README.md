# snn-me-bptt

## Reproducibility

Required packages: `timm, spikingjelly=0.0.0.0.15, torch>=2.0.0, torchvision`.
`triton` and `cupy` are also needed for reproducing the training speed experiments. However, they are not necessary for the memory experiments.

Overall instructions:

* `train.py` runs the full training process and measure time cost. `memory.py` profiles memory cost.
* "FGC" means sGC, "PGC" means pdsGC.
* Use the `-h` flag to explore other options (like `-lomo, -amp`, ...)
* Before running the scripts, fill in your wandb accounts to the scripts. Alternatively, you can run with `WANDB_MODE=disabled`.
* Only single-gpu training is supported now.

### Sequential CIFAR-10

```bash
python src/scifar/[train/memory].py --data_dir [path] --neuron_type [SJLIF/HandWrittenLIF/PSN/SlidingPSN] --spike_compressor [NullSpikeCompressor/BitSpikeCompressor] --network [SequentialCIFARNet/FGCSequentialCIFARNet/PGCSequentialCIFARNet] -nc 10 
```

### CIFAR10-DVS

```bash
python src/cifar10dvs/[train/memory].py --data_dir [path] --neuron_type [SJLIF/HandWrittenLIF/PSN/SlidingPSN] --spike_compressor [NullSpikeCompressor/BitSpikeCompressor] --network [CIFAR10DVSVGG/FGCCIFAR10DVSVGG/PGCCIFAR10DVSVGG]
```

### ImageNet, SEW ResNet

```bash
python src/imagenet/sew/[train/memory].py --data_dir [path] --neuron_type [SJLIF/HandWrittenLIF/PSN/SlidingPSN] --spike_compressor [NullSpikeCompressor/BitSpikeCompressor] --network [SEWResNet34/FGCSEWResNet34/PGCSEWResNet34]
```

### ImageNet, Transformer

```bash
python src/imagenet/transformer/[train/memory].py --data_dir [path] --neuron_type [SJLIF/HandWrittenLIF/PSN/SlidingPSN] --spike_compressor [NullSpikeCompressor/BitSpikeCompressor] --network [Spikformer/FGCSpikformer/PGCSpikformer/QKFormer/FGCQKFormer/PGCQKFormer]
```

## Known Issues

Q1: `Triton Error [CUDA]: device kernel image is invalid`
A1: According to [this comment](https://github.com/InternLM/lmdeploy/pull/1621#issuecomment-2179731554), the `ptxas` prepackaged in Triton is not compatible with our cuda driver version. We can specify the path to the correct `ptxas` by the environment variable `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.
