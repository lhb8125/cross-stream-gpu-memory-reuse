#!/usr/bin/env python3
"""Render the published microbenchmark summary as a standalone SVG."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


LABELS = {
    "same-stream/no-sync": "same-stream",
    "cross-stream/record-stream/no-sync": "record_stream",
    "cross-stream/record-stream/sync": "sync-each",
    "cross-stream/hand-back/no-sync": "hand-back",
    "cross-stream/ring1/no-sync": "slot pool K=1",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="CSV produced by summarize_results.py")
    parser.add_argument("--output", type=Path, default=Path("microbenchmark_tradeoff.svg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    rows = [row for row in rows if row["case"] in LABELS]
    if not rows:
        raise ValueError("input contains none of the five safe cases")

    width, height = 1280, 620
    left, bar_x = 72, 300
    reserved_width = 390
    host_x = 850
    host_width = 240
    row_y, row_step = 168, 82
    max_reserved = max(float(row["reserved_peak_mib"]) for row in rows)
    max_enqueue = max(float(row["enqueue_median_ms"]) for row in rows)
    elements = [
        f'<rect width="{width}" height="{height}" rx="28" fill="#081522"/>',
        '<g font-family="Inter,system-ui,-apple-system,sans-serif">',
        '<text x="72" y="64" fill="#edf7ff" font-size="30" font-weight="750">'
        "Cross-Stream Lifetime Strategies: Memory and Host Enqueue Time</text>",
        '<text x="72" y="96" fill="#9fb4c8" font-size="16">'
        "512 MiB BF16 input · 52 iterations · medians of five independent runs</text>",
        f'<text x="{bar_x}" y="132" fill="#52d8ff" font-size="15" font-weight="700">'
        "Peak reserved</text>",
        f'<text x="{host_x}" y="132" fill="#f4ca64" font-size="15" font-weight="700">'
        "Host enqueue</text>",
    ]

    for index, row in enumerate(rows):
        y = row_y + index * row_step
        reserved = float(row["reserved_peak_mib"])
        pending = float(row["pending_mib"])
        enqueue = float(row["enqueue_median_ms"])
        reserved_w = reserved_width * reserved / max_reserved
        pending_w = reserved_width * pending / max_reserved
        enqueue_w = host_width * enqueue / max_enqueue
        label = LABELS[row["case"]]
        elements.extend(
            [
                f'<text x="{left}" y="{y + 21}" fill="#edf7ff" font-size="17" '
                f'font-weight="650">{esc(label)}</text>',
                f'<rect x="{bar_x}" y="{y}" width="{reserved_width}" height="30" rx="8" fill="#10283a"/>',
                f'<rect x="{bar_x}" y="{y}" width="{reserved_w:.2f}" height="30" rx="8" '
                'fill="#52d8ff"/>',
                f'<rect x="{bar_x}" y="{y}" width="{pending_w:.2f}" height="30" rx="8" '
                'fill="#ff9a6c" opacity="0.95"/>',
                f'<text x="{bar_x + reserved_width + 12}" y="{y + 21}" fill="#c9d8e6" font-size="14">'
                f'{reserved:,.0f} MiB</text>',
                f'<rect x="{host_x}" y="{y}" width="{host_width}" height="30" rx="8" '
                'fill="#10283a"/>',
                f'<rect x="{host_x}" y="{y}" width="{enqueue_w:.2f}" height="30" '
                'rx="8" fill="#f4ca64"/>',
                f'<text x="{host_x + host_width + 12}" y="{y + 21}" fill="#c9d8e6" font-size="14">'
                f'{enqueue:.2f} ms</text>',
            ]
        )

    elements.extend(
        [
            '<rect x="72" y="574" width="18" height="12" rx="3" fill="#52d8ff"/>',
            '<text x="100" y="585" fill="#9fb4c8" font-size="14">reserved</text>',
            '<rect x="195" y="574" width="18" height="12" rx="3" fill="#ff9a6c"/>',
            '<text x="223" y="585" fill="#9fb4c8" font-size="14">pending-free subset</text>',
            '<text x="1160" y="585" text-anchor="end" fill="#9fb4c8" font-size="14">'
            "Source: data/microbenchmark_5x_summary.csv</text>",
            "</g>",
        ]
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="title desc">'
        '<title id="title">Cross-stream lifetime microbenchmark</title>'
        '<desc id="desc">Peak reserved and pending-free memory plus Host enqueue time '
        'for five safe tensor lifetime strategies.</desc>'
        + "".join(elements)
        + "</svg>\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
