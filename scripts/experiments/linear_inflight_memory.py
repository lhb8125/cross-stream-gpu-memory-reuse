#!/usr/bin/env python3
"""Measure allocator growth for a one-way cross-stream tensor lifetime.

The producer stream allocates and publishes a large tensor.  The consumer
stream waits for the producer and reads the tensor, but the producer has no
functional dependency on the consumer and may immediately enqueue the next
write.  This makes lifetime protection real rather than redundant:

* ``unsafe`` drops the last Python reference without protection and may reuse
  the input address while the consumer is still reading it.
* ``cross`` uses ``record_stream`` to preserve the one-way overlap safely.
* ``handback`` makes the producer wait for the last consumer before allowing
  dynamic storage reuse.
* ``ring`` keeps caller-owned input slots and guards each reuse with an event.

An independent GEMM backlog is queued before each input-consuming GEMM.  It
keeps host enqueue light while making an unsafe reuse easy to observe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


MIB = 1024**2


def allocator_state() -> dict[str, float | int]:
    """Return allocator counters without synchronizing CUDA."""
    stats = torch.cuda.memory_stats()
    states: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for segment in torch.cuda.memory_snapshot():
        for block in segment["blocks"]:
            state = block["state"]
            states[state][0] += int(block["size"])
            states[state][1] += 1

    # PyTorch snapshots have used both names across versions.  Keep the
    # public/docs name while accepting the internal build used by these runs.
    pending_bytes = states["active_awaiting_free"][0] + states["active_pending_free"][0]
    pending_blocks = states["active_awaiting_free"][1] + states["active_pending_free"][1]

    return {
        "allocated_current_mib": stats["allocated_bytes.all.current"] / MIB,
        "allocated_peak_mib": stats["allocated_bytes.all.peak"] / MIB,
        "active_current_mib": stats["active_bytes.all.current"] / MIB,
        "active_peak_mib": stats["active_bytes.all.peak"] / MIB,
        "reserved_current_mib": stats["reserved_bytes.all.current"] / MIB,
        "reserved_peak_mib": stats["reserved_bytes.all.peak"] / MIB,
        "active_allocated_mib": states["active_allocated"][0] / MIB,
        "active_allocated_blocks": states["active_allocated"][1],
        "active_awaiting_free_mib": pending_bytes / MIB,
        "active_awaiting_free_blocks": pending_blocks,
        "active_pending_free_mib": pending_bytes / MIB,
        "active_pending_free_blocks": pending_blocks,
        "inactive_mib": states["inactive"][0] / MIB,
        "inactive_blocks": states["inactive"][1],
        "num_device_alloc": stats.get("num_device_alloc", -1),
        "num_alloc_retries": stats.get("num_alloc_retries", -1),
    }


def make_done_event(stream: torch.cuda.Stream) -> torch.cuda.Event:
    event = torch.cuda.Event(enable_timing=False)
    event.record(stream)
    return event


def nvtx_push(enabled: bool, label: str) -> None:
    if enabled:
        torch.cuda.nvtx.range_push(label)


def nvtx_pop(enabled: bool) -> None:
    if enabled:
        torch.cuda.nvtx.range_pop()


def start_memory_history(args: argparse.Namespace) -> None:
    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="python",
            max_entries=args.history_max_entries,
        )


def dump_memory_history(args: argparse.Namespace) -> None:
    if not args.record_memory_history or not args.snapshot_path:
        return
    path = Path(args.snapshot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._dump_snapshot(str(path))


@torch.inference_mode()
def run_case(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    dtype = getattr(torch, args.dtype)
    element_bytes = torch.empty((), dtype=dtype).element_size()
    tensor_mib = args.m * args.k * element_bytes / MIB
    output_mib = args.m * args.n * element_bytes / MIB

    consumer_stream = torch.cuda.default_stream(device)
    producer_stream = consumer_stream if args.stream_mode == "same" else torch.cuda.Stream()

    # The input-consuming GEMM returns the iteration payload exactly: only the
    # first row of the weight is one, so every output element equals x[:, 0].
    weight = torch.zeros((args.k, args.n), device=device, dtype=dtype)
    weight[0].fill_(1)
    consumer_output = torch.empty((args.m, args.n), device=device, dtype=dtype)
    checksums = torch.empty(args.iters, device=device, dtype=torch.float32)

    # Independent GEMMs create a realistic consumer backlog before x is read.
    # Their output is persistent, so they do not add dynamic allocator noise.
    delay_a = torch.zeros(
        (args.consumer_delay_dim, args.consumer_delay_dim), device=device, dtype=dtype
    )
    delay_b = torch.zeros_like(delay_a)
    delay_output = torch.empty_like(delay_a)

    def enqueue_consumer_work(x: torch.Tensor, iteration: int) -> torch.cuda.Event:
        for _ in range(args.consumer_delay_repeats):
            torch.mm(delay_a, delay_b, out=delay_output)
        torch.mm(x, weight, out=consumer_output)
        checksums[iteration].copy_(consumer_output[0, 0].float())
        return make_done_event(consumer_stream)

    # Warm cuBLAS, stream/event creation, and allocator paths.
    warm_x = torch.empty((args.m, args.k), device=device, dtype=dtype)
    warm_x.fill_(1)
    enqueue_consumer_work(warm_x, 0)
    del warm_x
    torch.cuda.synchronize()

    input_ring: list[torch.Tensor] = []
    slot_last_consumer: list[torch.cuda.Event | None] = []
    if args.stream_mode == "ring":
        if args.slots <= 0:
            raise ValueError("--slots must be positive for --stream-mode ring")
        with torch.cuda.stream(producer_stream):
            input_ring = [
                torch.empty((args.m, args.k), device=device, dtype=dtype)
                for _ in range(args.slots)
            ]
        slot_last_consumer = [None] * args.slots
        torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_memory_history(args)
    baseline = allocator_state()

    done_events: list[torch.cuda.Event] = []
    input_addresses: list[int] = []
    start = time.perf_counter()

    for iteration in range(args.iters):
        nvtx_push(args.nvtx, f"{args.stream_mode}/iteration_{iteration}")
        payload = iteration + 1

        if args.stream_mode == "ring":
            slot = iteration % args.slots
            previous_consumer = slot_last_consumer[slot]
            if previous_consumer is not None:
                producer_stream.wait_event(previous_consumer)
            x = input_ring[slot]
        else:
            with torch.cuda.stream(producer_stream):
                x = torch.empty((args.m, args.k), device=device, dtype=dtype)

        input_addresses.append(x.data_ptr())
        with torch.cuda.stream(producer_stream):
            x.fill_(payload)
            ready = make_done_event(producer_stream)

        consumer_stream.wait_event(ready)
        with torch.cuda.stream(consumer_stream):
            done = enqueue_consumer_work(x, iteration)
            if args.stream_mode == "cross":
                # No consumer -> producer dependency exists.  The allocator
                # must therefore retain x until this stream has finished.
                x.record_stream(consumer_stream)

        done_events.append(done)
        if args.stream_mode == "handback":
            # GPU-side hand-back: future allocations on producer_stream are
            # ordered after the last consumer without blocking the host.
            producer_stream.wait_event(done)
        elif args.stream_mode == "ring":
            slot_last_consumer[slot] = done

        if args.stream_mode != "ring":
            del x
        if args.sync_each:
            torch.cuda.synchronize()
        nvtx_pop(args.nvtx)

    cpu_enqueue_s = time.perf_counter() - start
    completed_before_sync = sum(event.query() for event in done_events)
    before_sync = allocator_state()
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - start
    after_sync = allocator_state()
    dump_memory_history(args)

    observed = checksums.cpu()
    expected = torch.arange(1, args.iters + 1, dtype=torch.float32)
    output_finite = bool(torch.isfinite(observed).all().item())
    output_max_abs_error = float((observed - expected).abs().max().item())
    output_correct = output_finite and output_max_abs_error == 0.0
    if args.stream_mode != "unsafe" and not output_correct:
        raise RuntimeError(
            "numerical correctness check failed: "
            f"finite={output_finite}, max_abs_error={output_max_abs_error}"
        )

    device_alloc_delta = before_sync["num_device_alloc"] - baseline["num_device_alloc"]

    result = {
        "case": {
            "same": "same-stream",
            "unsafe": "cross-stream/unsafe-no-lifetime-protection",
            "cross": "cross-stream/record-stream",
            "handback": "cross-stream/hand-back",
            "ring": f"cross-stream/ring{args.slots}",
        }[args.stream_mode]
        + f"/{'sync' if args.sync_each else 'no-sync'}",
        "repeat_index": args.repeat_index,
        "shape": [args.m, args.k],
        "dtype": args.dtype,
        "iters": args.iters,
        "tensor_mib": tensor_mib,
        "consumer_output_mib": output_mib,
        "consumer_delay_dim": args.consumer_delay_dim,
        "consumer_delay_repeats": args.consumer_delay_repeats,
        "completed_before_final_sync": completed_before_sync,
        "inflight_before_final_sync": args.iters - completed_before_sync,
        "cpu_enqueue_s": cpu_enqueue_s,
        "wall_s": wall_s,
        "ring_slots": args.slots if args.stream_mode == "ring" else 0,
        "snapshot_path": args.snapshot_path,
        "device_alloc_delta": device_alloc_delta,
        "output_finite": output_finite,
        "output_correct": output_correct,
        "output_max_abs_error": output_max_abs_error,
        "unique_input_addresses": len(set(input_addresses)),
        "first_observed_values": observed[: min(8, args.iters)].tolist(),
        "baseline": baseline,
        "before_final_sync": before_sync,
        "after_final_sync": after_sync,
    }
    result_line = json.dumps(result, sort_keys=True)
    print(result_line, flush=True)
    if args.jsonl_output:
        output_path = Path(args.jsonl_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(result_line + "\n")


def run_all(args: argparse.Namespace) -> None:
    cases: list[tuple[str, bool, int]] = [
        ("same", False, 0),
        ("cross", False, 0),
        ("handback", False, 0),
    ]
    if args.extended_matrix:
        cases.insert(1, ("unsafe", False, 0))
        cases.insert(3, ("cross", True, 0))
        cases.extend(("ring", False, slots) for slots in args.ring_slots)

    if args.jsonl_output:
        output_path = Path(args.jsonl_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    for repeat_index in range(args.repeat):
        for stream_mode, sync_each, slots in cases:
            command = [
                sys.executable,
                __file__,
                "--case",
                "--stream-mode",
                stream_mode,
                "--m",
                str(args.m),
                "--k",
                str(args.k),
                "--n",
                str(args.n),
                "--iters",
                str(args.iters),
                "--consumer-delay-dim",
                str(args.consumer_delay_dim),
                "--consumer-delay-repeats",
                str(args.consumer_delay_repeats),
                "--dtype",
                args.dtype,
                "--repeat-index",
                str(repeat_index),
            ]
            if sync_each:
                command.append("--sync-each")
            if stream_mode == "ring":
                command.extend(("--slots", str(slots)))
            if args.nvtx:
                command.append("--nvtx")
            if args.jsonl_output:
                command.extend(("--jsonl-output", args.jsonl_output))
            subprocess.run(command, check=True)


def comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("slot counts must be positive")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="store_true", help="Run one case instead of the matrix")
    parser.add_argument(
        "--stream-mode",
        choices=("same", "unsafe", "cross", "handback", "ring"),
        default="same",
    )
    parser.add_argument("--sync-each", action="store_true")
    parser.add_argument("--slots", type=int, default=2, help="Persistent slots for ring mode")
    parser.add_argument(
        "--ring-slots",
        type=comma_separated_ints,
        default=(1, 2, 4),
        help="Comma-separated ring slot sweep used by the matrix (default: 1,2,4)",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the full case matrix")
    parser.add_argument(
        "--extended-matrix",
        action="store_true",
        help="Also run unsafe, sync-each, and ring slot controls",
    )
    parser.add_argument("--repeat-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--nvtx", action="store_true", help="Annotate iterations with NVTX")
    parser.add_argument(
        "--record-memory-history",
        action="store_true",
        help="Record allocator trace entries for an optional snapshot",
    )
    parser.add_argument("--snapshot-path", help="Dump memory snapshot after the final sync")
    parser.add_argument("--history-max-entries", type=int, default=200_000)
    parser.add_argument(
        "--jsonl-output",
        help="Write one machine-readable JSON record per case (matrix mode truncates first)",
    )
    parser.add_argument("--m", type=int, default=32768)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--iters", type=int, default=52)
    parser.add_argument("--consumer-delay-dim", type=int, default=4096)
    parser.add_argument("--consumer-delay-repeats", type=int, default=4)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.case:
        run_case(arguments)
    else:
        run_all(arguments)
