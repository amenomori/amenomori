# Yusen Gao

I work on AI infrastructure for training, especially RL post-training. My focus is distributed GPU execution, with particular attention to gradient communication and numerical precision.

## Distributed training

In verl, I exposed VeOmni's node-local ReduceScatter option through Hydra so training jobs can transfer gradients in BF16 and accumulate them in FP32. I also documented when to use it and its performance tradeoffs.

In VeOmni, I fixed a DPO initialization bug that occurred when the frozen reference model inherited the policy's Float8 training settings.

## GPU execution

In vLLM, I fixed a CUDA Graph padding mismatch during adaptive verification by making draft prefill use the same query-length bound as the target model across data-parallel ranks.
