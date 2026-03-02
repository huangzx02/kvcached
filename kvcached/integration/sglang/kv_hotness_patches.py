# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Runtime-gated KV hotness tracing patches for SGLang.

This module adds lightweight, exception-safe wrappers around ScheduleBatch
allocation paths to export sparse KV block-range hotness over time.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
import types
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch

from kvcached.cli.utils import get_hotness_control
from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.sglang.patches import SGLANG_ALL_RANGE
from kvcached.integration.version_utils import VersionAwarePatch, version_range
from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    if val in ("1", "true", "yes", "y", "on"):
        return True
    if val in ("0", "false", "no", "n", "off"):
        return False
    return default


def _feature_enabled() -> bool:
    return _env_flag("KVCACHED_SGLANG_KV_HOTNESS", False)


def _safe_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    return 0


def _sanitize_segment(segment: str) -> str:
    out: list[str] = []
    for ch in segment:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("-")
    return "".join(out)[:64]


def _to_int_list(x: Any) -> list[int]:
    if x is None:
        return []
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, tuple):
        return [int(v) for v in x]
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return []
        try:
            return [int(v) for v in x.detach().cpu().tolist()]
        except Exception:
            return [int(v) for v in x.to(dtype=torch.int64, copy=False).view(-1)]
    try:
        return [int(v) for v in x]
    except Exception:
        return []


def _aggregate_counts(
    bucket_ids: torch.Tensor,
    weights: Optional[torch.Tensor],
    out: Dict[int, int],
) -> None:
    if bucket_ids.numel() == 0:
        return
    if weights is None:
        uniq, counts = torch.unique(bucket_ids, return_counts=True)
        uniq_cpu = uniq.detach().cpu().tolist()
        counts_cpu = counts.detach().cpu().tolist()
        for bid, cnt in zip(uniq_cpu, counts_cpu):
            out[int(bid)] = int(out.get(int(bid), 0) + int(cnt))
        return

    uniq, inv = torch.unique(bucket_ids, return_inverse=True)
    sums = torch.zeros_like(uniq, dtype=torch.int64)
    sums.scatter_add_(0, inv, weights)
    uniq_cpu = uniq.detach().cpu().tolist()
    sums_cpu = sums.detach().cpu().tolist()
    for bid, cnt in zip(uniq_cpu, sums_cpu):
        out[int(bid)] = int(out.get(int(bid), 0) + int(cnt))


def compute_decode_weighted_bucket_counts(
    req_to_token: torch.Tensor,
    req_pool_indices: list[int],
    seq_lens: list[int],
    bucket_size: int,
) -> dict[int, int]:
    counts: dict[int, int] = {}
    if bucket_size <= 0:
        raise ValueError(f"bucket_size must be positive, got {bucket_size}")

    n = min(len(req_pool_indices), len(seq_lens))
    for i in range(n):
        req_idx = int(req_pool_indices[i])
        seq_len = int(seq_lens[i])
        if seq_len <= 0 or req_idx < 0:
            continue

        row = req_to_token[req_idx, :seq_len].to(dtype=torch.int64, copy=False)
        row = row[row >= 0]
        if row.numel() == 0:
            continue

        bucket_ids = torch.div(row, int(bucket_size), rounding_mode="floor")
        _aggregate_counts(bucket_ids, None, counts)

    return counts


def compute_extend_weighted_bucket_counts(
    req_to_token: torch.Tensor,
    req_pool_indices: list[int],
    seq_lens: list[int],
    extend_lens: list[int],
    bucket_size: int,
) -> dict[int, int]:
    counts: dict[int, int] = {}
    if bucket_size <= 0:
        raise ValueError(f"bucket_size must be positive, got {bucket_size}")

    n = min(len(req_pool_indices), len(seq_lens), len(extend_lens))
    for i in range(n):
        req_idx = int(req_pool_indices[i])
        seq_len = int(seq_lens[i])
        extend_len = int(extend_lens[i])
        if seq_len <= 0 or req_idx < 0:
            continue

        row = req_to_token[req_idx, :seq_len].to(dtype=torch.int64, copy=False)
        row = row[row >= 0]
        if row.numel() == 0:
            continue

        # Prefix tokens are revisited extend_len times in extend attention.
        prefix_len = max(0, seq_len - max(0, extend_len))
        prefix_len = min(prefix_len, int(row.numel()))

        if prefix_len > 0 and extend_len > 0:
            prefix_bucket_ids = torch.div(
                row[:prefix_len], int(bucket_size), rounding_mode="floor"
            )
            prefix_weights = torch.full(
                (prefix_bucket_ids.numel(),),
                fill_value=int(extend_len),
                dtype=torch.int64,
                device=prefix_bucket_ids.device,
            )
            _aggregate_counts(prefix_bucket_ids, prefix_weights, counts)

        # Tail token at position p contributes (seq_len - p).
        tail = row[prefix_len:]
        if tail.numel() > 0:
            tail_bucket_ids = torch.div(tail, int(bucket_size), rounding_mode="floor")
            tail_weights = torch.arange(
                int(tail.numel()),
                0,
                -1,
                dtype=torch.int64,
                device=tail_bucket_ids.device,
            )
            _aggregate_counts(tail_bucket_ids, tail_weights, counts)

    return counts


@dataclass
class _RecordState:
    enabled: bool = False
    session_id: int = 0
    revision: int = 0
    step_id: int = 0
    last_poll_ms: int = 0
    csv_fp: Any = None
    json_fp: Any = None
    csv_writer: Any = None


class _KVHotnessRecorder:
    def __init__(self) -> None:
        self.bucket_size = max(
            1, int(os.getenv("KVCACHED_SGLANG_KV_HOTNESS_BUCKET_SIZE", "64"))
        )
        self.sample_every = max(
            1, int(os.getenv("KVCACHED_SGLANG_KV_HOTNESS_SAMPLE_EVERY", "1"))
        )
        self.output_dir = os.getenv("KVCACHED_SGLANG_KV_HOTNESS_OUTPUT_DIR", "/tmp")
        self.file_prefix = os.getenv(
            "KVCACHED_SGLANG_KV_HOTNESS_FILE_PREFIX", "sglang_kv_hotness"
        )
        self.ctrl_poll_ms = max(
            50, int(os.getenv("KVCACHED_SGLANG_KV_HOTNESS_CTRL_POLL_MS", "500"))
        )
        self.rank0_only = _env_flag("KVCACHED_SGLANG_KV_HOTNESS_RANK0_ONLY", True)
        self._states: dict[str, _RecordState] = {}
        self._lock = threading.Lock()
        os.makedirs(self.output_dir, exist_ok=True)

    def _resolve_ipc_name(self, batch: Any) -> str:
        allocator = getattr(batch, "token_to_kv_pool_allocator", None)
        kvcached_allocator = getattr(allocator, "kvcached_allocator", None)
        page_allocator = getattr(kvcached_allocator, "page_allocator", None)
        mem_info_tracker = getattr(page_allocator, "mem_info_tracker", None)
        ipc_name = getattr(mem_info_tracker, "ipc_name", None)
        if isinstance(ipc_name, str) and ipc_name:
            return ipc_name

        fallback = os.getenv("KVCACHED_IPC_NAME")
        if fallback:
            return str(fallback)
        return "kvcached"

    def _open_session_files(self, state: _RecordState, ipc_name: str, rank: int) -> None:
        safe_ipc = _sanitize_segment(ipc_name)
        base = f"{self.file_prefix}_{safe_ipc}_rank{int(rank)}"
        csv_path = os.path.join(self.output_dir, f"{base}.csv")
        json_path = os.path.join(self.output_dir, f"{base}.jsonl")

        csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        state.csv_fp = open(csv_path, "a", newline="", buffering=1)
        state.csv_writer = csv.writer(state.csv_fp)
        if not csv_exists:
            state.csv_writer.writerow(
                [
                    "ts_unix_ms",
                    "rank",
                    "session_id",
                    "step_id",
                    "mode",
                    "bucket_size",
                    "bucket_id",
                    "bucket_start_block",
                    "bucket_end_block",
                    "weighted_count",
                    "num_reqs",
                    "num_tokens",
                ]
            )

        state.json_fp = open(json_path, "a", buffering=1)

    def _flush_close(self, state: _RecordState) -> None:
        try:
            if state.csv_fp is not None:
                state.csv_fp.flush()
                state.csv_fp.close()
        except Exception:
            pass
        try:
            if state.json_fp is not None:
                state.json_fp.flush()
                state.json_fp.close()
        except Exception:
            pass
        state.csv_fp = None
        state.csv_writer = None
        state.json_fp = None

    def _poll_control(self, ipc_name: str, state: _RecordState, now_ms: int, rank: int) -> None:
        if state.last_poll_ms != 0 and now_ms - state.last_poll_ms < self.ctrl_poll_ms:
            return

        state.last_poll_ms = now_ms
        try:
            control = get_hotness_control(ipc_name)
        except Exception as e:
            if state.enabled:
                logger.warning(f"Failed to read hotness control for {ipc_name}: {e}")
                self._flush_close(state)
            state.enabled = False
            return

        next_enabled = int(control.enabled) == 1
        next_session = int(control.session_id)
        next_revision = int(control.revision)

        # OFF transition: flush and close immediately.
        if state.enabled and not next_enabled:
            self._flush_close(state)

        # ON transition (or bumped session while ON): start a fresh step counter.
        if next_enabled and ((not state.enabled) or next_session != state.session_id):
            self._flush_close(state)
            state.step_id = 0
            self._open_session_files(state, ipc_name=ipc_name, rank=rank)

        state.enabled = next_enabled
        state.session_id = next_session
        state.revision = next_revision

    def _iter_csv_rows(
        self,
        *,
        ts_unix_ms: int,
        rank: int,
        session_id: int,
        step_id: int,
        mode: str,
        num_reqs: int,
        num_tokens: int,
        bucket_counts: dict[int, int],
    ) -> Iterable[list[int | str]]:
        for bucket_id, weighted_count in sorted(bucket_counts.items()):
            start_block = int(bucket_id) * int(self.bucket_size)
            end_block = start_block + int(self.bucket_size) - 1
            yield [
                int(ts_unix_ms),
                int(rank),
                int(session_id),
                int(step_id),
                str(mode),
                int(self.bucket_size),
                int(bucket_id),
                int(start_block),
                int(end_block),
                int(weighted_count),
                int(num_reqs),
                int(num_tokens),
            ]

    def record_batch(self, batch: Any, mode: str) -> None:
        if not _feature_enabled() or not enable_kvcached():
            return

        rank = _safe_rank()
        if self.rank0_only and rank != 0:
            return

        req_to_token_pool = getattr(batch, "req_to_token_pool", None)
        req_to_token = getattr(req_to_token_pool, "req_to_token", None)
        if not isinstance(req_to_token, torch.Tensor):
            return

        req_pool_indices = _to_int_list(getattr(batch, "req_pool_indices", None))
        if not req_pool_indices:
            return

        seq_lens = _to_int_list(getattr(batch, "seq_lens_cpu", None))
        if not seq_lens:
            seq_lens = _to_int_list(getattr(batch, "seq_lens", None))
        if not seq_lens:
            return

        if mode == "extend":
            extend_lens = _to_int_list(getattr(batch, "extend_lens", None))
            if not extend_lens:
                return
            bucket_counts = compute_extend_weighted_bucket_counts(
                req_to_token=req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                extend_lens=extend_lens,
                bucket_size=self.bucket_size,
            )
        else:
            bucket_counts = compute_decode_weighted_bucket_counts(
                req_to_token=req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                bucket_size=self.bucket_size,
            )

        num_reqs = int(min(len(req_pool_indices), len(seq_lens)))
        num_tokens = int(sum(int(x) for x in seq_lens[:num_reqs]))
        ts_unix_ms = int(time.time() * 1000)
        ipc_name = self._resolve_ipc_name(batch)
        state_key = f"{ipc_name}#{rank}"

        with self._lock:
            state = self._states.setdefault(state_key, _RecordState())
            self._poll_control(ipc_name=ipc_name, state=state, now_ms=ts_unix_ms, rank=rank)
            if not state.enabled:
                return

            step_id = int(state.step_id)
            state.step_id += 1
            if step_id % self.sample_every != 0:
                return

            if state.csv_writer is None or state.json_fp is None:
                self._open_session_files(state, ipc_name=ipc_name, rank=rank)

            assert state.csv_writer is not None
            assert state.json_fp is not None

            for row in self._iter_csv_rows(
                ts_unix_ms=ts_unix_ms,
                rank=rank,
                session_id=state.session_id,
                step_id=step_id,
                mode=mode,
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                bucket_counts=bucket_counts,
            ):
                state.csv_writer.writerow(row)

            json_record = {
                "ts_unix_ms": int(ts_unix_ms),
                "rank": int(rank),
                "session_id": int(state.session_id),
                "step_id": int(step_id),
                "mode": str(mode),
                "bucket_size": int(self.bucket_size),
                "num_reqs": int(num_reqs),
                "num_tokens": int(num_tokens),
                "buckets": [
                    [int(bucket_id), int(weighted_count)]
                    for bucket_id, weighted_count in sorted(bucket_counts.items())
                ],
            }
            state.json_fp.write(json.dumps(json_record, ensure_ascii=True, separators=(",", ":")) + "\n")
            state.csv_fp.flush()
            state.json_fp.flush()


_RECORDER = _KVHotnessRecorder()


class ScheduleBatchKVHotnessPatch(VersionAwarePatch, BasePatch):
    """Patch ScheduleBatch to export runtime-gated KV hotness traces."""

    library = "sglang"
    target_module = "sglang.srt.managers.schedule_batch"
    target_class = "ScheduleBatch"
    patch_name = "schedule_batch_kv_hotness"

    def apply(self, schedule_batch_mod: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False
        return self.patch_schedule_batch(schedule_batch_mod)

    @version_range(SGLANG_ALL_RANGE)
    def patch_schedule_batch(self, schedule_batch_mod: types.ModuleType) -> bool:
        ScheduleBatch = self._get_target_class(schedule_batch_mod)
        if ScheduleBatch is None:
            return False

        patched_any = False

        original_prepare_extend = getattr(ScheduleBatch, "prepare_for_extend", None)
        if original_prepare_extend is not None and not self._is_already_patched(
            original_prepare_extend
        ):

            def _wrapped_prepare_for_extend(self, *args: Any, **kwargs: Any):
                out = original_prepare_extend(self, *args, **kwargs)
                if _feature_enabled() and enable_kvcached():
                    try:
                        _RECORDER.record_batch(self, mode="extend")
                    except Exception:
                        pass
                return out

            self._mark_as_patched(_wrapped_prepare_for_extend)
            setattr(ScheduleBatch, "prepare_for_extend", _wrapped_prepare_for_extend)
            patched_any = True

        original_prepare_decode = getattr(ScheduleBatch, "prepare_for_decode", None)
        if original_prepare_decode is not None and not self._is_already_patched(
            original_prepare_decode
        ):

            def _wrapped_prepare_for_decode(self, *args: Any, **kwargs: Any):
                out = original_prepare_decode(self, *args, **kwargs)
                if _feature_enabled() and enable_kvcached():
                    try:
                        _RECORDER.record_batch(self, mode="decode")
                    except Exception:
                        pass
                return out

            self._mark_as_patched(_wrapped_prepare_for_decode)
            setattr(ScheduleBatch, "prepare_for_decode", _wrapped_prepare_for_decode)
            patched_any = True

        return patched_any
