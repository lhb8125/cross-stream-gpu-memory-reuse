#!/usr/bin/env python3
"""Aggregate repeated linear_inflight_memory JSONL records into CSV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SAFE_CASE_ORDER = (
    "same-stream/no-sync",
    "cross-stream/record-stream/no-sync",
    "cross-stream/record-stream/sync",
    "cross-stream/hand-back/no-sync",
    "cross-stream/ring1/no-sync",
)

FIELDS = (
    "case",
    "repeats",
    "allocated_peak_mib",
    "active_peak_mib",
    "reserved_peak_mib",
    "pending_mib",
    "pending_blocks",
    "inactive_mib",
    "device_alloc_delta",
    "unique_input_addresses",
    "completed_before_final_sync",
    "enqueue_median_ms",
    "wall_median_ms",
    "wall_cv_percent",
    "outputs_correct",
)


def median(records: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values: list[float] = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        values.append(float(value))
    return statistics.median(values)


def summarize(case: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    wall_ms = [float(record["wall_s"]) * 1000 for record in records]
    wall_mean = statistics.mean(wall_ms)
    wall_cv = statistics.pstdev(wall_ms) / wall_mean * 100 if wall_mean else 0.0
    return {
        "case": case,
        "repeats": len(records),
        "allocated_peak_mib": median(records, ("before_final_sync", "allocated_peak_mib")),
        "active_peak_mib": median(records, ("before_final_sync", "active_peak_mib")),
        "reserved_peak_mib": median(records, ("before_final_sync", "reserved_peak_mib")),
        "pending_mib": median(records, ("before_final_sync", "active_awaiting_free_mib")),
        "pending_blocks": median(
            records, ("before_final_sync", "active_awaiting_free_blocks")
        ),
        "inactive_mib": median(records, ("before_final_sync", "inactive_mib")),
        "device_alloc_delta": median(records, ("device_alloc_delta",)),
        "unique_input_addresses": median(records, ("unique_input_addresses",)),
        "completed_before_final_sync": median(records, ("completed_before_final_sync",)),
        "enqueue_median_ms": median(records, ("cpu_enqueue_s",)) * 1000,
        "wall_median_ms": statistics.median(wall_ms),
        "wall_cv_percent": wall_cv,
        "outputs_correct": all(bool(record["output_correct"]) for record in records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSONL emitted by the microbenchmark")
    parser.add_argument("--output", type=Path, default=Path("microbenchmark_summary.csv"))
    parser.add_argument(
        "--include-unsafe",
        action="store_true",
        help="Include the intentionally unsafe negative control",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
        grouped[str(record["case"])].append(record)

    order = list(SAFE_CASE_ORDER)
    if args.include_unsafe:
        order.insert(1, "cross-stream/unsafe-no-lifetime-protection/no-sync")
    missing = [case for case in SAFE_CASE_ORDER if case not in grouped]
    if missing:
        raise ValueError(f"missing safe cases: {', '.join(missing)}")

    rows = [summarize(case, grouped[case]) for case in order if case in grouped]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
