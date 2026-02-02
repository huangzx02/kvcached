# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Scheduler-specific SGLang patches.

These patches aim to make kvcached shrink "success-first" by proactively
retracting requests when shrink is blocked by locked KV pages.
"""

from __future__ import annotations

import types
from collections import defaultdict
from typing import Any, Iterable

import torch

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.version_utils import VersionAwarePatch, version_range
from kvcached.utils import get_kvcached_logger

from kvcached.integration.sglang.patches import SGLANG_ALL_RANGE

logger = get_kvcached_logger()


def _safe_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    return 0


def _get_allocated_by_page(kvcached_allocator: Any) -> dict[int, int]:
    allocated_by_page: dict[int, int] = {}
    for page in list(getattr(kvcached_allocator, "full_pages", {}).values()) + list(
        getattr(kvcached_allocator, "avail_pages", {}).values()
    ):
        num_kv_blocks = getattr(page, "num_kv_blocks", None)
        if num_kv_blocks is None:
            continue
        allocated = int(num_kv_blocks) - int(page.num_free_blocks())
        if allocated > 0:
            allocated_by_page[int(page.page_id)] = int(allocated)
    return allocated_by_page


def _iter_caches_for_allocator(kvcached_allocator: Any) -> Iterable[Any]:
    # Radix caches are registered via RadixCacheShrinkEvictionPatch.
    caches = getattr(kvcached_allocator, "_kvcached_radix_caches", None)
    if not caches:
        return []
    try:
        return list(caches)
    except Exception:
        return []


def _best_effort_evict_unlocked_pages(
    kvcached_allocator: Any, target_num_pages: int
) -> None:
    for cache in _iter_caches_for_allocator(kvcached_allocator):
        evict_fn = getattr(cache, "_kvcached_evict_pages_for_shrink", None)
        if not callable(evict_fn):
            continue
        try:
            evict_fn(int(target_num_pages))
        except Exception:
            # Best-effort.
            continue


def _choose_retract_indices_for_pages(
    batch: Any,
    kvcached_allocator: Any,
    target_num_pages: int,
) -> list[int]:
    """Choose request indices to retract to free whole pages first.

    This selection is page-aware (kvcached physical pages), and is designed for
    shrink success:
      - Prefer pages that become empty after retracting a small set of requests.
      - Tie-break by lower recompute cost (smaller kv_committed_len).
    """
    page_allocator = kvcached_allocator.page_allocator
    block_mem_size = int(kvcached_allocator.block_mem_size)
    page_size = int(page_allocator.page_size)

    inuse_pages = int(page_allocator.get_num_inuse_pages())
    if inuse_pages <= int(target_num_pages):
        return []

    pages_needed = inuse_pages - int(target_num_pages)
    allocated_by_page = _get_allocated_by_page(kvcached_allocator)
    if not allocated_by_page:
        return []

    excluded_pages: set[int] = set()
    null_block = getattr(kvcached_allocator, "null_block", None)
    if null_block:
        try:
            excluded_pages.add(
                int(page_allocator.get_page_id(int(null_block[0]), block_mem_size))
            )
        except Exception:
            pass

    reqs = list(getattr(batch, "reqs", []) or [])
    if len(reqs) <= 1:
        return []

    # page -> set(req_idx) to retract so that this page can be fully released:
    #   - retract frees per-request KV blocks beyond cache_protected_len
    #   - retract also unlocks radix nodes along the request path, allowing
    #     page-aware eviction to delete those nodes and free their blocks
    required_reqs_by_page: dict[int, set[int]] = defaultdict(set)

    # Directly freeable blocks on each page (freed by retract itself).
    freeable_blocks_by_page: dict[int, int] = defaultdict(int)

    req_cost: dict[int, int] = {}
    req_rid: dict[int, str] = {}

    req_to_token = batch.req_to_token_pool.req_to_token
    for i, req in enumerate(reqs):
        req_rid[i] = str(getattr(req, "rid", i))
        req_cost[i] = int(getattr(req, "kv_committed_len", 0) or 0)

        req_pool_idx = getattr(req, "req_pool_idx", None)
        if req_pool_idx is None:
            continue

        start = int(getattr(req, "cache_protected_len", 0) or 0)
        end = int(getattr(req, "kv_allocated_len", 0) or 0)
        if end <= start:
            continue

        try:
            idxs = req_to_token[req_pool_idx, start:end].to(dtype=torch.int64, copy=False)
        except Exception:
            continue
        if idxs.numel() == 0:
            continue

        pids = (idxs * block_mem_size) // page_size
        try:
            unique_pids, counts = torch.unique(pids, return_counts=True)
        except Exception:
            unique_pids = torch.unique(pids)
            counts = torch.ones_like(unique_pids, dtype=torch.int64)

        try:
            pids_list = unique_pids.detach().cpu().tolist()
            counts_list = counts.detach().cpu().tolist()
        except Exception:
            continue

        for pid, cnt in zip(pids_list, counts_list):
            pid_i = int(pid)
            if pid_i in excluded_pages:
                continue
            if pid_i not in allocated_by_page:
                continue
            required_reqs_by_page[pid_i].add(i)
            freeable_blocks_by_page[pid_i] += int(cnt)

    # Add requests required to unlock locked radix nodes on each page.
    tree_cache = getattr(batch, "tree_cache", None)
    root_node = getattr(tree_cache, "root_node", None) if tree_cache is not None else None
    if root_node is not None:
        # Map radix nodes to the request indices that lock them.
        node_id_to_node: dict[int, Any] = {}
        node_id_to_reqs: dict[int, set[int]] = defaultdict(set)
        for i, req in enumerate(reqs):
            node = getattr(req, "last_node", None)
            while node is not None and node is not root_node:
                nid = int(id(node))
                node_id_to_node[nid] = node
                node_id_to_reqs[nid].add(i)
                node = getattr(node, "parent", None)

        node_pages_cache: dict[int, list[int]] = {}
        for nid, req_set in node_id_to_reqs.items():
            node = node_id_to_node.get(nid)
            if node is None:
                continue
            if int(getattr(node, "lock_ref", 0) or 0) <= 0:
                continue
            val = getattr(node, "value", None)
            if val is None:
                continue

            pages = node_pages_cache.get(nid)
            if pages is None:
                try:
                    ids = val.to(dtype=torch.int64, copy=False)
                except Exception:
                    continue
                pids = (ids * block_mem_size) // page_size
                try:
                    unique_pids = torch.unique(pids)
                except Exception:
                    continue
                try:
                    pages = [int(x) for x in unique_pids.detach().cpu().tolist()]
                except Exception:
                    continue
                node_pages_cache[nid] = pages

            for pid_i in pages:
                if pid_i in excluded_pages:
                    continue
                if pid_i not in allocated_by_page:
                    continue
                required_reqs_by_page[pid_i].update(req_set)

    # Precompute per-page metadata for deterministic, low-overhead greedy selection.
    page_meta: dict[int, tuple[set[int], int, int, int, int, tuple[str, ...]]] = {}
    # Value: (req_set, allocated, freeable_blocks, can_free_direct, total_cost, rid_key)
    reqset_page_count: dict[frozenset[int], int] = defaultdict(int)
    for pid_i, req_set in required_reqs_by_page.items():
        if pid_i in excluded_pages:
            continue
        allocated = int(allocated_by_page.get(pid_i, 0))
        if allocated <= 0:
            continue
        freeable = int(freeable_blocks_by_page.get(pid_i, 0))
        can_free_direct = 1 if freeable >= allocated else 0
        total_cost = sum(int(req_cost.get(j, 0)) for j in req_set)
        rid_key = tuple(sorted(req_rid.get(j, str(j)) for j in req_set))
        page_meta[pid_i] = (set(req_set), allocated, freeable, can_free_direct, total_cost, rid_key)
        reqset_page_count[frozenset(req_set)] += 1

    if not page_meta:
        return []

    selected_pages: list[int] = []
    selected_reqs: set[int] = set()

    for _ in range(int(pages_needed)):
        best_key: tuple[int, int, int, int, int, int, int, tuple[str, ...]] | None = None
        best_pid: int | None = None
        best_req_set: set[int] | None = None

        for pid_i, (req_set, allocated, freeable, can_free_direct, total_cost, rid_key) in page_meta.items():
            if pid_i in selected_pages:
                continue
            additional = req_set - selected_reqs
            additional_count = len(additional)
            additional_cost = sum(int(req_cost.get(j, 0)) for j in additional)

            ratio_scaled = (freeable * 1_000_000) // max(int(allocated), 1)
            group_size = int(reqset_page_count.get(frozenset(req_set), 1))

            # Sort key:
            #   1) prefer pages that are freed by retract alone
            #   2) fewer *additional* retractions (greedy set cover)
            #   3) prefer request-sets that unlock more pages (cheap coverage heuristic)
            #   4) lower *additional* recompute cost
            #   5) fewer total retractions for this page
            #   6) higher direct-freeable ratio (less reliance on later evict)
            #   7) lower total recompute cost
            #   8) stable tie-breakers
            key = (
                -int(can_free_direct),
                int(additional_count),
                -int(group_size),
                int(additional_cost),
                int(len(req_set)),
                -int(ratio_scaled),
                int(total_cost),
                int(pid_i),
                rid_key,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_pid = int(pid_i)
                best_req_set = req_set

        if best_pid is None or best_req_set is None:
            break
        selected_pages.append(best_pid)
        selected_reqs.update(best_req_set)

    if not selected_pages or not selected_reqs:
        return []

    return sorted(selected_reqs, key=lambda j: req_rid.get(j, str(j)))


def _retract_indices_from_batch(
    scheduler: Any,
    batch: Any,
    retract_indices: list[int],
) -> Any:
    if not retract_indices:
        return batch

    reqs = list(getattr(batch, "reqs", []) or [])
    if not reqs:
        return batch

    retracted_reqs: list[Any] = []
    keep_set = set(range(len(reqs)))
    for idx in retract_indices:
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= len(reqs):
            continue
        if idx_i not in keep_set:
            continue
        keep_set.remove(idx_i)
        # Keep-set size is the remaining number of requests after this retraction.
        try:
            batch.release_req(idx_i, len(keep_set), scheduler.server_args)
            retracted_reqs.append(reqs[idx_i])
        except Exception:
            keep_set.add(idx_i)
            continue

    batch.filter_batch(keep_indices=sorted(keep_set))
    batch.batch_is_full = False

    # Put retracted requests back to the scheduler queue.
    for req in retracted_reqs:
        try:
            scheduler._add_request_to_queue(req, is_retracted=True)
        except Exception:
            pass

    return batch


class SchedulerShrinkRetractPatch(VersionAwarePatch, BasePatch):
    """Proactively retract requests to make kvcached shrink succeed."""

    library = "sglang"
    target_module = "sglang.srt.managers.scheduler"
    target_class = "Scheduler"
    patch_name = "scheduler_shrink_retract"

    def apply(self, sched_mod: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False
        return self.patch_scheduler(sched_mod)

    @version_range(SGLANG_ALL_RANGE)
    def patch_scheduler(self, sched_mod: types.ModuleType) -> bool:
        Scheduler = self._get_target_class(sched_mod)
        if Scheduler is None:
            return False

        patched_any = False

        original_get_next = getattr(Scheduler, "get_next_batch_to_run", None)
        if original_get_next is not None and not self._is_already_patched(original_get_next):

            def _wrapped_get_next_batch_to_run(self, *args: Any, **kwargs: Any):
                # Apply external resize target at the scheduler boundary so the
                # system reacts quickly to shrink requests (even before OOM).
                if enable_kvcached():
                    try:
                        tree_cache = getattr(self, "tree_cache", None)
                        allocator = (
                            getattr(tree_cache, "token_to_kv_pool_allocator", None)
                            if tree_cache
                            else None
                        )
                        kvcached_allocator = (
                            getattr(allocator, "kvcached_allocator", None)
                            if allocator
                            else None
                        )
                        apply_fn = getattr(kvcached_allocator, "maybe_apply_resize_target", None)
                        if callable(apply_fn):
                            apply_fn()
                    except Exception:
                        pass
                return original_get_next(self, *args, **kwargs)

            self._mark_as_patched(_wrapped_get_next_batch_to_run)
            setattr(Scheduler, "get_next_batch_to_run", _wrapped_get_next_batch_to_run)
            patched_any = True

        original_get_new_batch_prefill = getattr(Scheduler, "get_new_batch_prefill", None)
        if original_get_new_batch_prefill is not None and not self._is_already_patched(
            original_get_new_batch_prefill
        ):

            def _wrapped_get_new_batch_prefill(self, *args: Any, **kwargs: Any):
                if not enable_kvcached():
                    return original_get_new_batch_prefill(self, *args, **kwargs)

                # In SGLang, prefill allocation happens inside get_new_batch_prefill()
                # (via prepare_for_extend -> alloc). If shrink is pending, we must
                # finish it (or at least free enough pages) before that alloc.
                try:
                    tree_cache = getattr(self, "tree_cache", None)
                    allocator = (
                        getattr(tree_cache, "token_to_kv_pool_allocator", None)
                        if tree_cache
                        else None
                    )
                    kvcached_allocator = (
                        getattr(allocator, "kvcached_allocator", None) if allocator else None
                    )
                    if kvcached_allocator is not None:
                        try:
                            kvcached_allocator.maybe_apply_resize_target()
                        except Exception:
                            pass

                        if getattr(kvcached_allocator, "in_shrink", False) and getattr(
                            kvcached_allocator, "target_num_blocks", None
                        ):
                            target_num_blocks = int(kvcached_allocator.target_num_blocks)
                            block_mem_size = int(kvcached_allocator.block_mem_size)
                            new_mem_size = int(target_num_blocks * block_mem_size)
                            page_allocator = kvcached_allocator.page_allocator
                            target_num_pages = int(
                                new_mem_size // int(page_allocator.page_size)
                            )

                            _best_effort_evict_unlocked_pages(
                                kvcached_allocator, target_num_pages
                            )

                            if int(page_allocator.get_num_inuse_pages()) > target_num_pages:
                                running_batch = getattr(self, "running_batch", None)
                                if running_batch is not None and not running_batch.is_empty():
                                    retract_indices = _choose_retract_indices_for_pages(
                                        running_batch, kvcached_allocator, target_num_pages
                                    )
                                    if retract_indices:
                                        self.running_batch = _retract_indices_from_batch(
                                            self, running_batch, retract_indices
                                        )
                                        _best_effort_evict_unlocked_pages(
                                            kvcached_allocator, target_num_pages
                                        )
                                        try:
                                            kvcached_allocator.resize(new_mem_size)
                                        except Exception:
                                            pass

                            if _safe_rank() == 0:
                                running_bs = (
                                    self.running_batch.batch_size()
                                    if getattr(self, "running_batch", None) is not None
                                    else 0
                                )
                                logger.debug(
                                    "[shrink-retract/prefill] "
                                    f"inuse_pages={page_allocator.get_num_inuse_pages()} "
                                    f"target_pages={target_num_pages} "
                                    f"running_bs={running_bs}"
                                )
                except Exception:
                    pass

                return original_get_new_batch_prefill(self, *args, **kwargs)

            self._mark_as_patched(_wrapped_get_new_batch_prefill)
            setattr(Scheduler, "get_new_batch_prefill", _wrapped_get_new_batch_prefill)
            patched_any = True

        original = getattr(Scheduler, "update_running_batch", None)
        if original is None:
            return patched_any
        if self._is_already_patched(original):
            return patched_any or True

        def _wrapped(self, batch, *args: Any, **kwargs: Any):
            if not enable_kvcached() or batch is None or batch.is_empty():
                return original(self, batch, *args, **kwargs)

            tree_cache = getattr(batch, "tree_cache", None) or getattr(self, "tree_cache", None)
            allocator = getattr(tree_cache, "token_to_kv_pool_allocator", None) if tree_cache else None
            kvcached_allocator = getattr(allocator, "kvcached_allocator", None) if allocator else None
            if kvcached_allocator is None:
                return original(self, batch, *args, **kwargs)

            # Apply external resize target early (before mem checks / retraction).
            try:
                kvcached_allocator.maybe_apply_resize_target()
            except Exception:
                pass

            # If shrinking is still pending, attempt to unlock/free whole pages by
            # retracting selected requests, then re-run page-aware eviction.
            if getattr(kvcached_allocator, "in_shrink", False) and getattr(
                kvcached_allocator, "target_num_blocks", None
            ):
                target_num_blocks = int(kvcached_allocator.target_num_blocks)
                block_mem_size = int(kvcached_allocator.block_mem_size)
                new_mem_size = int(target_num_blocks * block_mem_size)
                page_allocator = kvcached_allocator.page_allocator
                target_num_pages = int(new_mem_size // int(page_allocator.page_size))

                # Best-effort eviction of already-unlocked radix nodes/pages.
                _best_effort_evict_unlocked_pages(kvcached_allocator, target_num_pages)

                if int(page_allocator.get_num_inuse_pages()) > target_num_pages:
                    retract_indices = _choose_retract_indices_for_pages(
                        batch, kvcached_allocator, target_num_pages
                    )
                    if retract_indices:
                        batch = _retract_indices_from_batch(self, batch, retract_indices)

                        # Evict again after unlocking to maximize whole-page frees.
                        _best_effort_evict_unlocked_pages(
                            kvcached_allocator, target_num_pages
                        )

                        # Try to finish shrink immediately if enough pages are freed.
                        try:
                            kvcached_allocator.resize(new_mem_size)
                        except Exception:
                            pass

                if _safe_rank() == 0:
                    logger.debug(
                        "[shrink-retract] "
                        f"inuse_pages={page_allocator.get_num_inuse_pages()} "
                        f"target_pages={target_num_pages} "
                        f"batch_size={batch.batch_size()}"
                    )

            return original(self, batch, *args, **kwargs)

        self._mark_as_patched(_wrapped)
        setattr(Scheduler, "update_running_batch", _wrapped)
        return True
