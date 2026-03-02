#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Plot SGLang KV hotness from sparse raw CSV traces.

This script aggregates per-step sparse bucket hotness into fixed tumbling windows,
computes per-window means, and exports both machine-readable CSV and heatmaps.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


@dataclass(frozen=True)
class HotnessRow:
    ts_unix_ms: int
    rank: int
    session_id: int
    step_id: int
    mode: str
    bucket_size: int
    bucket_id: int
    bucket_start_block: int
    bucket_end_block: int
    weighted_count: int
    num_reqs: int
    num_tokens: int


def _load_rows(csv_path: str) -> List[HotnessRow]:
    rows: list[HotnessRow] = []
    with open(csv_path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for raw in reader:
            try:
                rows.append(
                    HotnessRow(
                        ts_unix_ms=int(raw["ts_unix_ms"]),
                        rank=int(raw["rank"]),
                        session_id=int(raw["session_id"]),
                        step_id=int(raw["step_id"]),
                        mode=str(raw["mode"]),
                        bucket_size=int(raw["bucket_size"]),
                        bucket_id=int(raw["bucket_id"]),
                        bucket_start_block=int(raw["bucket_start_block"]),
                        bucket_end_block=int(raw["bucket_end_block"]),
                        weighted_count=int(raw["weighted_count"]),
                        num_reqs=int(raw["num_reqs"]),
                        num_tokens=int(raw["num_tokens"]),
                    )
                )
            except Exception:
                continue
    return rows


def _resolve_session(rows: List[HotnessRow], session_arg: str) -> int:
    sessions = sorted({r.session_id for r in rows})
    if not sessions:
        raise ValueError("No session IDs found in input rows")
    if session_arg == "latest":
        return int(sessions[-1])
    return int(session_arg)


def _aggregate_tumbling_window(
    rows: List[HotnessRow],
    window_ms: int,
) -> tuple[
    dict[tuple[int, int], float],
    dict[int, int],
    dict[int, tuple[int, int]],
    list[int],
    int,
]:
    if not rows:
        raise ValueError("No rows to aggregate")

    session_start_ms = min(r.ts_unix_ms for r in rows)

    # Sum counts by (window, bucket).
    total_count_by_wb: dict[tuple[int, int], int] = defaultdict(int)
    # Unique step keys per window for denominator.
    steps_by_window: dict[int, set[tuple[int, int]]] = defaultdict(set)
    # Bucket range metadata.
    bucket_ranges: dict[int, tuple[int, int]] = {}

    for r in rows:
        window_idx = int((r.ts_unix_ms - session_start_ms) // window_ms)
        total_count_by_wb[(window_idx, r.bucket_id)] += int(r.weighted_count)
        steps_by_window[window_idx].add((int(r.rank), int(r.step_id)))
        bucket_ranges[r.bucket_id] = (int(r.bucket_start_block), int(r.bucket_end_block))

    avg_by_wb: dict[tuple[int, int], float] = {}
    bucket_total_avg: dict[int, float] = defaultdict(float)
    for (window_idx, bucket_id), total in total_count_by_wb.items():
        steps_in_window = max(1, len(steps_by_window[window_idx]))
        avg = float(total) / float(steps_in_window)
        avg_by_wb[(window_idx, bucket_id)] = avg
        bucket_total_avg[bucket_id] += avg

    return (
        avg_by_wb,
        {w: len(s) for w, s in steps_by_window.items()},
        bucket_ranges,
        sorted(steps_by_window.keys()),
        session_start_ms,
    )


def _write_window_avg_csv(
    out_csv: str,
    *,
    avg_by_wb: dict[tuple[int, int], float],
    steps_in_window: dict[int, int],
    bucket_ranges: dict[int, tuple[int, int]],
    session_id: int,
    rank_label: str,
    window_ms: int,
    session_start_ms: int,
) -> None:
    with open(out_csv, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "session_id",
                "rank",
                "window_idx",
                "window_start_ms",
                "window_end_ms",
                "bucket_id",
                "bucket_start_block",
                "bucket_end_block",
                "steps_in_window",
                "avg_hotness",
            ]
        )

        for (window_idx, bucket_id), avg in sorted(avg_by_wb.items()):
            start_block, end_block = bucket_ranges[bucket_id]
            window_start_ms = session_start_ms + int(window_idx) * int(window_ms)
            window_end_ms = window_start_ms + int(window_ms)
            writer.writerow(
                [
                    int(session_id),
                    rank_label,
                    int(window_idx),
                    int(window_start_ms),
                    int(window_end_ms),
                    int(bucket_id),
                    int(start_block),
                    int(end_block),
                    int(steps_in_window.get(window_idx, 0)),
                    float(avg),
                ]
            )


def _build_matrix(
    *,
    avg_by_wb: dict[tuple[int, int], float],
    window_indices: list[int],
    top_buckets: int,
) -> tuple[np.ndarray, list[int]]:
    ordered_buckets = sorted({int(b) for (_w, b) in avg_by_wb.keys()})
    if top_buckets > 0:
        ordered_buckets = ordered_buckets[:top_buckets]

    window_to_col = {w: i for i, w in enumerate(window_indices)}
    bucket_to_row = {b: i for i, b in enumerate(ordered_buckets)}

    mat = np.zeros((len(ordered_buckets), len(window_indices)), dtype=np.float64)
    for (w, b), v in avg_by_wb.items():
        if b not in bucket_to_row:
            continue
        mat[bucket_to_row[b], window_to_col[w]] = float(v)

    return mat, ordered_buckets


def _build_time_window_labels(window_indices: list[int], window_ms: int) -> list[str]:
    labels: list[str] = []
    for w in window_indices:
        start_s = (int(w) * int(window_ms)) / 1000.0
        end_s = ((int(w) + 1) * int(window_ms)) / 1000.0
        labels.append(f"[{start_s:.2f},{end_s:.2f})s")
    return labels


def _compute_vmax(matrix: np.ndarray, vmax_percentile: Optional[float]) -> Optional[float]:
    if vmax_percentile is None:
        return None
    if not np.isfinite(float(vmax_percentile)):
        return None

    p = min(100.0, max(0.0, float(vmax_percentile)))
    vals = matrix[np.isfinite(matrix) & (matrix > 0)]
    if vals.size == 0:
        vals = matrix[np.isfinite(matrix)]
    if vals.size == 0:
        return None

    vmax = float(np.percentile(vals, p))
    if not np.isfinite(vmax):
        return None
    return vmax


def _compute_axis_stats(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Time-axis stats: per window over buckets.
    time_mean = np.mean(matrix, axis=0)
    time_var = np.var(matrix, axis=0)
    # Space-axis stats: per bucket over windows.
    space_mean = np.mean(matrix, axis=1)
    space_var = np.var(matrix, axis=1)
    return time_mean, time_var, space_mean, space_var


def _positive_or_nan(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.where(arr > 0, arr, np.nan)


def _set_log_limits(ax, values: np.ndarray, axis: str) -> None:
    vals = np.asarray(values, dtype=np.float64)
    pos = vals[np.isfinite(vals) & (vals > 0)]
    if pos.size == 0:
        if axis == "x":
            ax.set_xlim(1e-12, 1.0)
        else:
            ax.set_ylim(1e-12, 1.0)
        return
    lo = float(np.min(pos))
    hi = float(np.max(pos))
    if hi <= lo:
        hi = lo * (1.0 + 1e-6)
    lo = lo / 1.6
    hi = hi * 1.6
    if axis == "x":
        ax.set_xlim(lo, hi)
    else:
        ax.set_ylim(lo, hi)


def _plot_png(
    out_png: str,
    *,
    matrix: np.ndarray,
    window_indices: list[int],
    window_labels: list[str],
    ordered_buckets: list[int],
    bucket_size: int,
    window_sec: float,
    session_id: int,
    rank_label: str,
    dpi: int,
    zero_color: Optional[str],
    vmax_percentile: Optional[float],
    log_color: bool,
) -> None:
    fig_w = max(12, min(28, 8 + len(window_indices) * 0.42))
    fig_h = max(8, min(30, 6 + len(ordered_buckets) * 0.10))

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.4, 4.6],
        width_ratios=[4.8, 1.9],
        hspace=0.04,
        wspace=0.05,
    )
    ax_time = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_time)
    ax_space = fig.add_subplot(gs[1, 1], sharey=ax_heat)
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis("off")

    time_mean, time_var, space_mean, space_var = _compute_axis_stats(matrix)
    time_mean_plot = _positive_or_nan(time_mean)
    time_var_plot = _positive_or_nan(time_var)
    space_mean_plot = _positive_or_nan(space_mean)
    space_var_plot = _positive_or_nan(space_var)
    x_pos = np.arange(len(window_indices), dtype=np.int32)
    y_pos = np.arange(len(ordered_buckets), dtype=np.int32)

    vmax = _compute_vmax(matrix, vmax_percentile)
    cmap = plt.get_cmap("viridis").copy() if zero_color else plt.get_cmap("viridis")
    if zero_color:
        cmap.set_bad(color=zero_color)

    if log_color:
        positive_vals = matrix[matrix > 0]
        if positive_vals.size == 0:
            raise RuntimeError("Cannot use --log-color: no positive hotness values found.")
        vmin = float(np.min(positive_vals))
        vmax_eff = float(np.max(positive_vals)) if vmax is None else float(vmax)
        if vmax_eff <= vmin:
            vmax_eff = vmin * (1.0 + 1e-6)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax_eff)
        if zero_color:
            plot_data = np.ma.masked_where(matrix <= 0, matrix)
        else:
            plot_data = np.where(matrix <= 0, vmin, matrix)
        im = ax_heat.imshow(
            plot_data,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
    else:
        if zero_color:
            plot_data = np.ma.masked_where(matrix == 0, matrix)
        else:
            plot_data = matrix
        im = ax_heat.imshow(
            plot_data,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )

    ax_heat.set_title(
        f"SGLang KV Hotness Tumbling Window Mean (window={window_sec:.2f}s, session={session_id}, rank={rank_label})"
    )
    ax_heat.set_xlabel("Time window (fixed tumbling windows, relative to session start)")
    ax_heat.set_ylabel(f"Bucket ID (ascending, bucket size={bucket_size})")

    ax_time_var = ax_time.twinx()
    line_time_mean = ax_time.plot(
        x_pos, time_mean_plot, color="tab:blue", linewidth=1.8, label="Mean"
    )[0]
    line_time_var = ax_time_var.plot(
        x_pos, time_var_plot, color="tab:orange", linewidth=1.8, label="Variance"
    )[0]
    ax_time.set_yscale("log")
    ax_time_var.set_yscale("log")
    _set_log_limits(ax_time, time_mean_plot, axis="y")
    _set_log_limits(ax_time_var, time_var_plot, axis="y")
    ax_time.set_ylabel("Mean Hotness (log)")
    ax_time_var.set_ylabel("Variance Hotness (log)")
    ax_time.grid(True, alpha=0.25)
    ax_time.legend(
        [line_time_mean, line_time_var], ["Mean", "Variance"], loc="upper right", fontsize=8
    )
    ax_time.set_title("Time-Axis Mean/Variance (dual log-y, aligned with heatmap x-axis)")
    ax_time.tick_params(axis="x", labelbottom=False)

    if len(window_indices) <= 30:
        tick_idx = np.arange(len(window_indices))
    else:
        tick_step = max(1, len(window_indices) // 20)
        tick_idx = np.arange(0, len(window_indices), tick_step)
    ax_heat.set_xticks(tick_idx)
    ax_heat.set_xticklabels([window_labels[i] for i in tick_idx], rotation=45, ha="right")

    if len(ordered_buckets) <= 40:
        ax_heat.set_yticks(y_pos)
        ax_heat.set_yticklabels([str(b) for b in ordered_buckets])

    ax_space_var = ax_space.twiny()
    line_space_mean = ax_space.plot(
        space_mean_plot, y_pos, color="tab:blue", linewidth=1.8, label="Mean"
    )[0]
    line_space_var = ax_space_var.plot(
        space_var_plot, y_pos, color="tab:orange", linewidth=1.8, label="Variance"
    )[0]
    ax_space.set_xscale("log")
    ax_space_var.set_xscale("log")
    _set_log_limits(ax_space, space_mean_plot, axis="x")
    _set_log_limits(ax_space_var, space_var_plot, axis="x")
    ax_space.set_xlabel("Mean Hotness (log)")
    ax_space_var.set_xlabel("Variance Hotness (log)")
    ax_space_var.xaxis.set_label_position("top")
    ax_space_var.xaxis.set_ticks_position("top")
    ax_space.grid(True, alpha=0.25)
    ax_space_var.grid(False)
    ax_space.tick_params(axis="y", labelleft=False)
    ax_space.set_title("Space-Axis Mean/Variance\n(dual log-x, aligned with heatmap y-axis)")
    ax_space.legend(
        [line_space_mean, line_space_var], ["Mean", "Variance"], loc="lower right", fontsize=8
    )

    cbar = fig.colorbar(im, ax=ax_heat, pad=0.012, fraction=0.05)
    label = "Average weighted accesses per scheduler step"
    if log_color:
        label += " (log scale)"
    cbar.set_label(label)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot_html(
    out_html: str,
    *,
    matrix: np.ndarray,
    window_labels: list[str],
    ordered_buckets: list[int],
    bucket_size: int,
    window_sec: float,
    session_id: int,
    rank_label: str,
    zero_color: Optional[str],
    vmax_percentile: Optional[float],
    log_color: bool,
) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        print("Warning: plotly is not available; skipped HTML heatmap output.")
        return False

    time_mean, time_var, space_mean, space_var = _compute_axis_stats(matrix)
    time_mean_plot = _positive_or_nan(time_mean)
    time_var_plot = _positive_or_nan(time_var)
    space_mean_plot = _positive_or_nan(space_mean)
    space_var_plot = _positive_or_nan(space_var)
    x_pos = np.arange(len(window_labels), dtype=np.int32)
    y_pos = np.arange(len(ordered_buckets), dtype=np.int32)

    vmax = _compute_vmax(matrix, vmax_percentile)
    colorbar_title = "Average weighted accesses per scheduler step"

    if log_color:
        positive_vals = matrix[matrix > 0]
        if positive_vals.size == 0:
            print("Warning: no positive values for --log-color; falling back to linear color.")
            log_color = False
        else:
            vmin = float(np.min(positive_vals))
            vmax_eff = float(np.max(positive_vals)) if vmax is None else float(vmax)
            if vmax_eff <= vmin:
                vmax_eff = vmin * (1.0 + 1e-6)

            if zero_color:
                z_data = np.where(matrix <= 0, np.nan, np.log10(matrix))
            else:
                z_data = np.log10(np.where(matrix <= 0, vmin, matrix))

            zmin = float(np.log10(vmin))
            zmax = float(np.log10(vmax_eff))
            if zmax <= zmin:
                zmax = zmin + 1e-6

            tick_vals = np.linspace(zmin, zmax, num=6).tolist()
            tick_text = [f"{10**v:.2f}" for v in tick_vals]
            colorbar_title += " (log scale)"
    if not log_color:
        zmin = 0.0
        zmax = vmax
        if zero_color:
            z_data = np.where(matrix == 0, np.nan, matrix)
        else:
            z_data = matrix
        tick_vals = None
        tick_text = None

    if zero_color:
        hover_on_gaps = False
    else:
        hover_on_gaps = True

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.24, 0.76],
        column_widths=[0.79, 0.21],
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
        specs=[
            [{"type": "xy", "secondary_y": True}, {"type": "xy"}],
            [{"type": "heatmap"}, {"type": "xy"}],
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=x_pos,
            y=time_mean_plot,
            mode="lines",
            name="Time Mean",
            line={"color": "#1f77b4", "width": 2},
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x_pos,
            y=time_var_plot,
            mode="lines",
            name="Time Variance",
            line={"color": "#ff7f0e", "width": 2},
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=x_pos,
            y=y_pos,
            colorscale="Viridis",
            zmin=zmin,
            zmax=zmax,
            colorbar={
                "title": colorbar_title,
                "x": 0.785,
                "len": 0.72,
                **({"tickvals": tick_vals, "ticktext": tick_text} if tick_vals is not None else {}),
            },
            hoverongaps=hover_on_gaps,
            hovertemplate=(
                "window=%{x}<br>"
                "bucket_row=%{y}<br>"
                "avg_hotness=%{z}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=space_mean_plot,
            y=y_pos,
            mode="lines",
            name="Space Mean",
            line={"color": "#1f77b4", "width": 2},
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=space_var_plot,
            y=y_pos,
            mode="lines",
            name="Space Variance",
            line={"color": "#ff7f0e", "width": 2},
            xaxis="x5",
            yaxis="y4",
        ),
    )

    fig.update_xaxes(matches="x3", row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(type="log", title_text="Time Mean (log)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(type="log", title_text="Time Variance (log)", row=1, col=1, secondary_y=True)

    if len(window_labels) <= 30:
        tick_idx = np.arange(len(window_labels))
    else:
        tick_step = max(1, len(window_labels) // 20)
        tick_idx = np.arange(0, len(window_labels), tick_step)
    fig.update_xaxes(
        row=2,
        col=1,
        title_text="Time window (fixed tumbling windows, relative to session start)",
        tickmode="array",
        tickvals=tick_idx.tolist(),
        ticktext=[window_labels[i] for i in tick_idx],
        tickangle=45,
    )

    if len(ordered_buckets) <= 40:
        fig.update_yaxes(
            row=2,
            col=1,
            tickmode="array",
            tickvals=y_pos.tolist(),
            ticktext=[str(b) for b in ordered_buckets],
        )
    fig.update_yaxes(
        row=2,
        col=1,
        title_text=f"Bucket ID (ascending, bucket size={bucket_size})",
        autorange="reversed",
    )

    fig.update_yaxes(matches="y3", row=2, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    fig.update_xaxes(type="log", title_text="Space Mean (log)", row=2, col=2)

    x4_domain = fig.layout.xaxis4.domain
    fig.update_layout(
        xaxis5={
            "domain": x4_domain,
            "anchor": "y4",
            "overlaying": "x4",
            "side": "top",
            "type": "log",
            "title": {"text": "Space Variance (log)"},
            "showgrid": False,
        }
    )

    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)

    layout_kwargs = {
        "title": f"SGLang KV Hotness Tumbling Window Mean (window={window_sec:.2f}s, session={session_id}, rank={rank_label})",
        "legend": {"orientation": "h", "x": 0.02, "y": 1.02},
    }
    if zero_color:
        layout_kwargs["plot_bgcolor"] = zero_color
    fig.update_layout(**layout_kwargs)
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SGLang KV hotness heatmaps from raw sparse CSV")
    parser.add_argument("--input-csv", required=True, help="Raw hotness CSV path")
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output file prefix (without suffix); defaults to input path stem",
    )
    parser.add_argument("--window-sec", type=float, default=3.0, help="Fixed tumbling window size in seconds")
    parser.add_argument("--session", default="latest", help="Session ID to plot or 'latest'")
    parser.add_argument("--rank", type=int, default=None, help="Optional rank filter")
    parser.add_argument(
        "--top-buckets",
        type=int,
        default=0,
        help="Maximum number of buckets to keep after bucket_id ascending sort (0 means all)",
    )
    parser.add_argument(
        "--zero-color",
        default=None,
        help="Optional color for cells with avg hotness == 0 (PNG and HTML)",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=None,
        help="Optional upper color limit percentile over positive hotness values (e.g., 99)",
    )
    parser.add_argument(
        "--log-color",
        action="store_true",
        help="Use logarithmic color scale for heatmaps",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    rows = _load_rows(args.input_csv)
    if args.rank is not None:
        rows = [r for r in rows if int(r.rank) == int(args.rank)]

    if not rows:
        raise RuntimeError("No rows matched the selected input/session/rank")

    session_id = _resolve_session(rows, str(args.session))
    rows = [r for r in rows if int(r.session_id) == int(session_id)]
    if not rows:
        raise RuntimeError(f"No rows found for session {session_id}")

    bucket_sizes = sorted({int(r.bucket_size) for r in rows})
    if len(bucket_sizes) != 1:
        raise RuntimeError(f"Expected a single bucket_size, got {bucket_sizes}")
    bucket_size = int(bucket_sizes[0])

    window_ms = max(1, int(float(args.window_sec) * 1000))
    avg_by_wb, steps_in_window, bucket_ranges, window_indices, session_start_ms = _aggregate_tumbling_window(
        rows,
        window_ms,
    )

    matrix, ordered_buckets = _build_matrix(
        avg_by_wb=avg_by_wb,
        window_indices=window_indices,
        top_buckets=max(0, int(args.top_buckets)),
    )
    window_labels = _build_time_window_labels(window_indices, window_ms)
    if matrix.size == 0:
        raise RuntimeError("No non-zero bucket data available after filtering/top-k")

    if args.out_prefix:
        out_prefix = args.out_prefix
    else:
        stem, _ = os.path.splitext(args.input_csv)
        out_prefix = stem

    out_csv = f"{out_prefix}_window_avg.csv"
    out_png = f"{out_prefix}_heatmap.png"
    out_html = f"{out_prefix}_heatmap.html"

    rank_label = str(args.rank) if args.rank is not None else "all"

    _write_window_avg_csv(
        out_csv,
        avg_by_wb=avg_by_wb,
        steps_in_window=steps_in_window,
        bucket_ranges=bucket_ranges,
        session_id=session_id,
        rank_label=rank_label,
        window_ms=window_ms,
        session_start_ms=session_start_ms,
    )

    _plot_png(
        out_png,
        matrix=matrix,
        window_indices=window_indices,
        window_labels=window_labels,
        ordered_buckets=ordered_buckets,
        bucket_size=bucket_size,
        window_sec=float(args.window_sec),
        session_id=session_id,
        rank_label=rank_label,
        dpi=int(args.dpi),
        zero_color=args.zero_color,
        vmax_percentile=args.vmax_percentile,
        log_color=bool(args.log_color),
    )

    _plot_html(
        out_html,
        matrix=matrix,
        window_labels=window_labels,
        ordered_buckets=ordered_buckets,
        bucket_size=bucket_size,
        window_sec=float(args.window_sec),
        session_id=session_id,
        rank_label=rank_label,
        zero_color=args.zero_color,
        vmax_percentile=args.vmax_percentile,
        log_color=bool(args.log_color),
    )

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_png}")
    print(f"Attempted: {out_html}")


if __name__ == "__main__":
    main()
