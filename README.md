# Cross-Stream GPU Memory Reuse

Reproduction material for the technical article *Why Does Cross-Stream GPU
Memory Keep Growing?* The experiment isolates a one-way CUDA stream lifetime:
a producer publishes a tensor, a consumer reads it on another stream, and the
Host can enqueue new producer work before the consumer retires.

The repository compares five safe lifetime strategies and one intentionally
unsafe control:

- same-stream ordering;
- cross-stream use protected by `Tensor.record_stream()`;
- `record_stream()` plus a Host synchronization on every iteration;
- creation-stream hand-back with a consumer-to-producer CUDA event;
- an event-managed preallocated slot pool;
- an unsafe cross-stream control used only to verify the race detector.

## Repository contents

```text
.
├── data/
│   ├── evidence.json                 # Numbers used by the article
│   └── microbenchmark_5x_summary.csv # Published five-run summary
├── figures/
│   └── microbenchmark_tradeoff.svg   # Rebuilt from the published CSV
└── scripts/
    ├── experiments/
    │   └── linear_inflight_memory.py # CUDA microbenchmark and snapshot export
    ├── plot_microbenchmark.py        # Dependency-free SVG renderer
    └── summarize_results.py          # JSONL-to-CSV aggregation
```

## Requirements

- Linux with an NVIDIA GPU
- Python 3.10 or newer
- CUDA-enabled PyTorch
- enough free GPU memory for the selected shape

The published H100 measurements used PyTorch
`2.11.0a0+a6c236b9fd.nv26.03.46836102` and CUDA 13.2. The default input shape
is `[32768, 8192]` in BF16, or 512 MiB per iteration. The `record_stream()`
case can retain roughly 22 GiB of pending-free storage, so use a smaller shape
for a smoke test on a lower-memory GPU.

Install PyTorch using the command appropriate for your CUDA environment. No
third-party package is required by the aggregation or plotting scripts.

## Reproduce the five safe cases

The unsafe case also runs as a negative control but is excluded from the safe
summary by default.

```bash
python scripts/linear_inflight_memory.py \
  --m 32768 --k 8192 --iters 52 --repeat 5 \
  --extended-matrix --ring-slots 1 \
  --jsonl-output one_way_five_safe_cases_5x.jsonl

python scripts/summarize_results.py \
  one_way_five_safe_cases_5x.jsonl \
  --output microbenchmark_5x_summary.csv

python scripts/plot_microbenchmark.py \
  microbenchmark_5x_summary.csv \
  --output microbenchmark_tradeoff.svg
```

For a lower-memory smoke test:

```bash
python scripts/linear_inflight_memory.py \
  --m 4096 --k 2048 --n 128 --iters 8 --repeat 1 \
  --consumer-delay-dim 1024 --consumer-delay-repeats 2 \
  --extended-matrix --ring-slots 1 \
  --jsonl-output smoke.jsonl
```

The smaller case checks that the program runs, but it may not create a deep
enough consumer backlog to reproduce the published pending-free magnitude or
make the unsafe race deterministic.

## Capture Memory Snapshots

Record the `record_stream()` path:

```bash
python scripts/linear_inflight_memory.py \
  --case --stream-mode cross --m 32768 --k 8192 --iters 52 \
  --nvtx --record-memory-history \
  --snapshot-path cross_record_stream.pickle
```

Record the creation-stream hand-back path:

```bash
python scripts/linear_inflight_memory.py \
  --case --stream-mode handback --m 32768 --k 8192 --iters 52 \
  --nvtx --record-memory-history \
  --snapshot-path handback.pickle
```

Snapshot files are generated locally rather than committed: they can be large
and can contain local Python source paths in recorded stack frames. Inspect
them with the PyTorch Memory Snapshot viewer or another compatible tool.

## Reading the result

The key comparison is not `allocated_peak_mib` alone. Inspect these fields
together:

- `active_awaiting_free_mib`: storage whose Python reference is gone but whose
  side-stream consumer has not yet retired;
- `reserved_peak_mib`: the allocator high-water mark;
- `completed_before_final_sync`: how far the GPU progressed while the Host was
  enqueueing work;
- `cpu_enqueue_s` versus `wall_s`: whether the Host remained ahead of the GPU;
- `output_correct` and `output_max_abs_error`: the numerical safety check.

The checked-in `data/evidence.json` also contains the production-training,
slot-pool, CUDA Graph, and long-tail values cited by the article. Those results
require the corresponding Megatron-LM environment and are included as
structured evidence, not as a claim that this standalone microbenchmark can
reproduce a full training run.

## References

- [PyTorch `Tensor.record_stream()`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)
- [PyTorch CUDA memory statistics](https://pytorch.org/docs/stable/generated/torch.cuda.memory.memory_stats.html)
- [Understanding CUDA Memory Usage](https://pytorch.org/docs/stable/torch_cuda_memory.html)
- [NVIDIA CUDA Graph Memory Issues](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/memory-issues.html)
- [Megatron-LM PR #7062](https://github.com/NVIDIA/Megatron-LM/pull/7062)
