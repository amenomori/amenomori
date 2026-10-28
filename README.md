# Yusen Gao

I work on AI infrastructure for training, especially RL post-training. My focus is distributed GPU execution, with particular attention to gradient communication and numerical precision.

## Distributed training

In [verl](https://github.com/verl-project/verl), I connected VeOmni's node-local low-precision ReduceScatter support to Hydra configuration and the model builder. Training jobs can transfer gradients in BF16 while accumulating in FP32. I also documented the requirements and performance tradeoffs.

## GPU execution

I also work on inference execution in [vLLM](https://github.com/vllm-project/vllm), particularly speculative decoding and CUDA Graph dispatch. I fixed a draft-prefill padding mismatch with the target's shared attention metadata during adaptive verification. The fix reuses the synchronized query-length bound across data-parallel ranks, with regression tests for cases where another rank is prefilling or has a longer decode query.
