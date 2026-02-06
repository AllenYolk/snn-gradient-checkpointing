# snn-gradient-checkpointing

The official implementation of [Towards Lossless Memory-efficient Training of Spiking Neural Networks via Gradient Checkpointing and Spike Compression](https://openreview.net/forum?id=nrBJ0Uvj7c) (ICLR 2026) by Yifan Huang, Wei Fang, Zecheng Hao, Zhengyu Ma and Yonghong Tian.

This work is an extension and augmentation of [SpikingJelly](https://github.com/fangwei123456/spikingjelly), conceived and developed by SpikingJelly's developer group. The `memory_optimization` API and its tutorial will soon be available in the latest version of SpikingJelly.

## Environment Setup

Dependencies are specified in `pyproject.toml`. We suggest using [uv](https://docs.astral.sh/uv/) for environment setup and package management.

1. Install `uv` according to the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
2. Create a new virtual environment locally: `uv venv`.
3. Install the dependencies: `uv sync --extra cu118`. Here, `cu118` specifies the CUDA version for PyTorch; other available options are `cpu`, `cu126`, and `cu128`.

Experienced developers can also manually install the specified dependencies.

## Reproduce the Results

### File Structure

Experiment scripts can be found in `src/<experiment_name>`.

* `src/cifar10dvs`: CIFAR10-DVS
* `src/dvsgesture`: DVS128 Gesture
* `src/imagenet/sew`: ImageNet, using SEW ResNet
* `src/imagenet/transformer`: ImageNet, using Spikformer or QKFormer; SpikeVideoFormer for Kinetics is also located in this directory.
* `src/scifar`: Sequential CIFAR-10 (and Sequential CIFAR-100)
* `src/shd`: Spiking Heidelberg Digits

In each experiment directory, the following scripts can be found:

* `models.py`: SNN definition
* `train.py`: training script; accuracies, training speed and peak memory usage will be printed to stdout
* `config.yaml`: configuration file that `LightningCLI` reads

### Run the Experiments

We organize the code after `lightning`'s style, using `LightningCLI` as the commandline interface. For a better understanding of our code, we strongly recommend you to read [lightning's tutorial and docs](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) first.

Take Sequential CIFAR-10 as an example. Use the `--help` flag to see all available CLI arguments.

```shell
> python src/scifar/train.py --help

......
usage: train.py [-c CONFIG] [--seed_everything SEED_EVERYTHING] [--trainer CONFIG] [--trainer.accelerator ACCELERATOR] [--trainer.strategy STRATEGY] [--trainer.devices DEVICES]
                [--trainer.num_nodes NUM_NODES] [--trainer.precision PRECISION] [--trainer.logger LOGGER] [--trainer.callbacks CALLBACKS] [--trainer.fast_dev_run FAST_DEV_RUN]
                [--trainer.max_epochs MAX_EPOCHS] [--trainer.min_epochs MIN_EPOCHS] [--trainer.max_steps MAX_STEPS] [--trainer.min_steps MIN_STEPS] [--trainer.max_time MAX_TIME]
                [--trainer.limit_train_batches LIMIT_TRAIN_BATCHES] [--trainer.limit_val_batches LIMIT_VAL_BATCHES] [--trainer.limit_test_batches LIMIT_TEST_BATCHES]
                [--trainer.limit_predict_batches LIMIT_PREDICT_BATCHES] [--trainer.overfit_batches OVERFIT_BATCHES] [--trainer.val_check_interval VAL_CHECK_INTERVAL]
                [--trainer.check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--trainer.num_sanity_val_steps NUM_SANITY_VAL_STEPS] [--trainer.log_every_n_steps LOG_EVERY_N_STEPS]
                [--trainer.enable_checkpointing {true,false,null}] [--trainer.enable_progress_bar {true,false,null}] [--trainer.enable_model_summary {true,false,null}]
                [--trainer.accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--trainer.gradient_clip_val GRADIENT_CLIP_VAL]
                [--trainer.gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM] [--trainer.deterministic DETERMINISTIC] [--trainer.benchmark {true,false,null}]
                [--trainer.inference_mode {true,false}] [--trainer.use_distributed_sampler {true,false}] [--trainer.profiler PROFILER] [--trainer.detect_anomaly {true,false}]
                [--trainer.barebones {true,false}] [--trainer.plugins PLUGINS] [--trainer.sync_batchnorm {true,false}]
                [--trainer.reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS] [--trainer.default_root_dir DEFAULT_ROOT_DIR]
                [--trainer.enable_autolog_hparams {true,false}] [--trainer.model_registry MODEL_REGISTRY] [--model CONFIG] --model.channels CHANNELS
                --model.neuron_type NEURON_TYPE --model.num_classes NUM_CLASSES --model.compress_x {true,false} --model.level LEVEL --model.decay_lambda DECAY_LAMBDA
                --model.learning_rate LEARNING_RATE --model.momentum MOMENTUM [--model.lomo {true,false}] [--data CONFIG] --data.data_dir DATA_DIR
                [--data.num_classes NUM_CLASSES] [--data.batch_size BATCH_SIZE] [--data.num_workers NUM_WORKERS] [--optimizer CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE]
                [--lr_scheduler CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE]
```

The arguments' default values are listed in `config.yaml`. Critical arguments include:

* `--config` or `-c`: should point to the configuration YAML file.
* `--trainer.accelerator`: typically set as `gpu`
* `--trainer.devices`: an integer indicating the total number of used devices (`2` means using 2 GPUs), or a list of GPU indices (`"[2]"` means using GPU 2; `"[1,2]"` means using GPU 1 and 2).
* `--model.neuron_type`: spiking neuron model (`SJLIF`, `PTLIF`, `MELIF`, `PSN`, and `SlidingPSN`)
* `--model.compress_x`: whether to use spike compression.
* `--model.level`: optimization level, ranging from 0 to 4.

For instance, you can run Sequential CIFAR-10 experiments with the following command:

```shell
python src/scifar/train.py --config src/scifar/config.yaml --trainer.accelerator gpu --trainer.devices "[2]" --model.neuron_type MELIF --model.compress_x True --model.level 4
```

For other CLI arguments, see the `config.yaml` files. Do not set those default CLI arguments provided by `LightningCLI` unless you fully understand the outcome!

### Other Experiments

Usage of these scripts are similar to the above.

* `src/cifar10dvs/memory_profile.py`: layer-wise memory usage profiling
* `src/cifar10dvs/time_profile.py`: layer-wise forward / backward pass time cost profiling
* `src/cifar10dvs/train_first_l.py`: apply GC only to the first `L` targeted blocks

## Use the Pipeline for Your Own SNN

The `memory_optimization` API will soon be available in the latest version of [SpikingJelly](https://github.com/fangwei123456/spikingjelly). Read the docs and tutorials for more details.

## Q&A

Q1: `Triton Error [CUDA]: device kernel image is invalid`
A1: According to [this comment](https://github.com/InternLM/lmdeploy/pull/1621#issuecomment-2179731554), the `ptxas` prepackaged in Triton is not compatible with your cuda driver version. You can specify the path to the correct `ptxas` by the environment variable `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.

Q2: The latest spikingjelly package cannot be cloned from GitHub when running `uv sync`.
A2: Try the OpenI mirror instead. Comment out line 68 of `pyproject.toml`, and uncomment line 69.

Q3: Package resolution failed.
A3: Delete `uv.lock`, and run `uv sync` again.

## Citation

If you find our work and code helpful, please consider citing our paper:

```bibtex
@inproceedings{huang2026towards,
    title={Towards Lossless Memory-efficient Training of Spiking Neural Networks via Gradient Checkpointing and Spike Compression},
    author={Yifan Huang and Wei Fang and Zecheng Hao and Zhengyu Ma and Yonghong Tian},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=nrBJ0Uvj7c}
}
```
