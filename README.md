# snn-me-bptt

TODO: Add README docs.

## Known Issues

Q1: `Triton Error [CUDA]: device kernel image is invalid`
A1: According to [this comment](https://github.com/InternLM/lmdeploy/pull/1621#issuecomment-2179731554), the `ptxas` prepackaged in Triton is not compatible with our cuda driver version. We can specify the path to the correct `ptxas` by the environment variable `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.
