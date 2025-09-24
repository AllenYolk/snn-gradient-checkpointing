# snn-me-bptt

## Preparation

Required packages: `timm, spikingjelly=0.0.0.0.15, torch>=2.0.0, triton, torchvision, lightning`.

* We use torch 2.7.1, triton 3.3.1 for the experiments.
* To install the latest version of `spikingjelly`, please clone the [github repository](https://github.com/fangwei123456/spikingjelly) and run `pip install .` in the its root directory.

## Reproduce the Results

Experiment scripts can be found in `src/<dataset_name>`.

* `src/scifar`: Sequential CIFAR-10 (and Sequential CIFAR-100)
* `src/cifar10dvs`: CIFAR10-DVS
* `src/dvsgesture`: DVS128 Gesture
* `src/shd`: SHD
* `src/imagenet/sew`: ImageNet, using SEW ResNet
* `src/imagenet/transformer`: ImageNet, using Spikformer or QKFormer; SpikeVideoFormer for Kinetics is also located in this directory.

In each experiment directory, the following scripts can be found:

* `models.py`: SNN definitions
* `train.py`: training script; accuracies, training speed and peak memory usage will be printed to stdout
* `config.yaml`: configuration file that `LightningCLI` reads

We organize the code after `lightning`'s style, using `LightningCLI` as the commandline interface. For a better understanding of our code, we strongly recommend you to read [lightning's tutorial and docs](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) first. Use the `--help` flag to see all available CLI arguments.

```shell
> python src/scifar/train.py --help

......
usage: train.py [-h] [-c CONFIG] [--print_config[=flags]] [--seed_everything SEED_EVERYTHING] [--trainer CONFIG] [--trainer.accelerator.help CLASS_PATH_OR_NAME]
                [--trainer.accelerator ACCELERATOR] [--trainer.strategy.help CLASS_PATH_OR_NAME] [--trainer.strategy STRATEGY] [--trainer.devices DEVICES]
                [--trainer.num_nodes NUM_NODES] [--trainer.precision PRECISION] [--trainer.logger.help CLASS_PATH_OR_NAME] [--trainer.logger LOGGER]
                [--trainer.callbacks.help CLASS_PATH_OR_NAME] [--trainer.callbacks CALLBACKS] [--trainer.fast_dev_run FAST_DEV_RUN] [--trainer.max_epochs MAX_EPOCHS]
                [--trainer.min_epochs MIN_EPOCHS] [--trainer.max_steps MAX_STEPS] [--trainer.min_steps MIN_STEPS] [--trainer.max_time MAX_TIME]
                [--trainer.limit_train_batches LIMIT_TRAIN_BATCHES] [--trainer.limit_val_batches LIMIT_VAL_BATCHES] [--trainer.limit_test_batches LIMIT_TEST_BATCHES]
                [--trainer.limit_predict_batches LIMIT_PREDICT_BATCHES] [--trainer.overfit_batches OVERFIT_BATCHES] [--trainer.val_check_interval VAL_CHECK_INTERVAL]
                [--trainer.check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--trainer.num_sanity_val_steps NUM_SANITY_VAL_STEPS] [--trainer.log_every_n_steps LOG_EVERY_N_STEPS]
                [--trainer.enable_checkpointing {true,false,null}] [--trainer.enable_progress_bar {true,false,null}] [--trainer.enable_model_summary {true,false,null}]
                [--trainer.accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--trainer.gradient_clip_val GRADIENT_CLIP_VAL]
                [--trainer.gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM] [--trainer.deterministic DETERMINISTIC] [--trainer.benchmark {true,false,null}]
                [--trainer.inference_mode {true,false}] [--trainer.use_distributed_sampler {true,false}] [--trainer.profiler.help CLASS_PATH_OR_NAME] [--trainer.profiler PROFILER]
                [--trainer.detect_anomaly {true,false}] [--trainer.barebones {true,false}] [--trainer.plugins.help CLASS_PATH_OR_NAME] [--trainer.plugins PLUGINS]
                [--trainer.sync_batchnorm {true,false}] [--trainer.reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS]
                [--trainer.default_root_dir DEFAULT_ROOT_DIR] [--trainer.model_registry MODEL_REGISTRY] [--model CONFIG] --model.channels CHANNELS
                --model.neuron_type NEURON_TYPE --model.num_classes NUM_CLASSES --model.compress_x {true,false} --model.level LEVEL --model.decay_lambda DECAY_LAMBDA
                --model.learning_rate LEARNING_RATE --model.momentum MOMENTUM [--model.lomo {true,false}] [--data CONFIG] --data.data_dir DATA_DIR
                [--data.num_classes NUM_CLASSES] [--data.batch_size BATCH_SIZE] [--data.num_workers NUM_WORKERS] [--optimizer.help [CLASS_PATH_OR_NAME]]
                [--optimizer CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE] [--lr_scheduler.help CLASS_PATH_OR_NAME]
                [--lr_scheduler CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE]

Lightning Trainer command line tool

options:
......
```

The arguments' default values are listed in `config.yaml`. Critical arguments include:

* `--config`: should point to the configuration YAML file.
* `--trainer.accelerator`: typically set as `gpu`
* `--trainer.devices`: an integer indicating the total number of used devices (`2` means using 2 GPUs), or a list of GPU indices (`"[2]"` means using GPU 2; `"[1,2]"` means using GPU 1 and 2). We suggest using the later style.
* `--model.neuron_type`: spiking neuron model. `SJLIF`, `HandWrittenLIF` (a.k.a. MELIF in the manuscript), `PSN`, and `SlidingPSN` are supported.
* `--model.compress_x`: whether to use spike compression.
* `--model.level`: optimization level, ranging from 0 to 4.

For instance, you may run Sequential CIFAR-10 experiments with the following command:

```shell
python src/scifar/train.py --config ./src/scifar/config.yaml --model.neuron_type HandWrittenLIF --model.level 4 --trainer.accelerator gpu --trainer.devices "[2]"
```

For other CLI arguments, see the `config.yaml` files. Do not set those default CLI arguments provided by `LightningCLI` unless you fully understand their meanings!

## Use the Framework on Your Own SNNs

The user interface `memory_optimization` is defined in `src/modules/checkpointing.py`. A brief tutorial is provided in Appendix I.

## Known Issues

Q1: `Triton Error [CUDA]: device kernel image is invalid`
A1: According to [this comment](https://github.com/InternLM/lmdeploy/pull/1621#issuecomment-2179731554), the `ptxas` prepackaged in Triton is not compatible with our cuda driver version. We can specify the path to the correct `ptxas` by the environment variable `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.

Q2: Is multi-device training possible?
A2: Yes. The framework is compatible with DDP. To enable DDP, use `--trainer.devices` to specify multiple devices (e.g. `--trainer.devices "[0,1,2]"`).
