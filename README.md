# Yusen Gao

I work on RL training infrastructure and LLM inference, particularly distributed execution, low-precision communication and GPU correctness.

## Open source

### [verl](https://github.com/verl-project/verl)

I connected VeOmni's node-local low-precision ReduceScatter support to verl's Hydra configuration and model builder. The change is merged. It enables BF16 gradient transport with FP32 accumulation, and I documented the requirements and performance tradeoffs.

### [vLLM](https://github.com/vllm-project/vllm)

My work in vLLM focuses on speculative decoding and CUDA Graph dispatch. I have a fix under review that keeps draft-prefill padding consistent with the target's shared attention metadata during adaptive verification. It reuses the synchronized query-length bound across data-parallel ranks, with regression tests for cases where another rank is prefilling or has a longer decode query.

I also fixed the setup of a DFlash CPU regression test. The test now runs the existing post-load hook before its first forward pass, with its dtype and output checks unchanged. That fix is merged.

I use reproducible bug reports and regression tests to check changes, and keep benchmark claims tied to the workloads measured.

<sub>Contribution status checked on October 28, 2026.</sub>
