# Yusen Gao

I work on RL training infrastructure and LLM inference, particularly distributed execution, low-precision communication and GPU correctness.

## Open source

### [verl](https://github.com/verl-project/verl)

I connected VeOmni's node-local low-precision ReduceScatter support to verl's Hydra configuration and model builder. It enables BF16 gradient transport with FP32 accumulation, and I documented the requirements and performance tradeoffs.

### [vLLM](https://github.com/vllm-project/vllm)

My work in vLLM focuses on speculative decoding and CUDA Graph dispatch. I fixed a draft-prefill padding mismatch with the target's shared attention metadata during adaptive verification. The fix reuses the synchronized query-length bound across data-parallel ranks, with regression tests for cases where another rank is prefilling or has a longer decode query.
