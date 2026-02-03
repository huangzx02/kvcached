"""
Radix cache-specific SGLang patches.

These patches aim to make kvcached shrink "success-first" by proactively
evicting radix cache entries to help shrink succeed.
"""

from __future__ import annotations

import time
import weakref
from collections import defaultdict
import heapq
from typing import Any

import torch

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.version_utils import VersionAwarePatch, version_range
from kvcached.integration.sglang.patches import SGLANG_ALL_RANGE
from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()

def _kvcached_evict_pages_for_shrink_impl(cache, target_num_pages: int) -> int:
    """Evict radix cache entries (not in compute) to help shrink succeed.

    Goal: free whole kvcached physical pages to satisfy shrink.

    Strategy (lock-aware + page-aware):
      1) Identify "fully releasable" pages: pages that are fully covered by radix
         cache nodes and have no blocks belonging to locked nodes (lock_ref > 0).
      2) If there are more fully releasable pages than needed, choose the subset
         with the lowest eviction cost (estimated by subtree token count).
         Otherwise, evict all fully releasable pages.
      3) For each selected page, evict the nodes that touch that page and all of
         their descendants (subtree deletion) by repeatedly deleting leaves until
         selected pages become empty.

    Notes:
      - This routine is best-effort and time-bounded.
      - It only deletes nodes with lock_ref == 0.
      - Determinism matters for TP: all ordering uses block-id-derived keys.
    """

    def _safe_rank() -> int:
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return int(torch.distributed.get_rank())
        except Exception:
            pass
        return 0

    if getattr(cache, "disable", False):
        return 0

    allocator = getattr(getattr(cache, "token_to_kv_pool_allocator", None), "kvcached_allocator", None)
    if allocator is None:
        return 0

    page_allocator = allocator.page_allocator
    if getattr(cache, "page_size", 1) != 1:
        # The kvcached SGLang integration is only validated with page_size=1.
        return 0

    block_mem_size = allocator.block_mem_size
    rank = _safe_rank()

    max_rounds = 5
    max_time_ms = 1000.0
    start_time = time.perf_counter()

    def _time_exceeded() -> bool:
        return (time.perf_counter() - start_time) * 1000 >= max_time_ms

    def _stable_child_key(x: Any) -> tuple[str, str]:
        # Keys may be custom objects; keep sort stable without relying on __lt__.
        return (type(x).__name__, repr(x))

    def _get_allocated_by_page() -> dict[int, int]:
        allocated_by_page: dict[int, int] = {}
        for page in list(allocator.full_pages.values()) + list(allocator.avail_pages.values()):
            if page.num_kv_blocks is None:
                continue
            allocated = int(page.num_kv_blocks) - int(page.num_free_blocks())
            if allocated > 0:
                allocated_by_page[int(page.page_id)] = int(allocated)
        return allocated_by_page

    def _iter_nodes_postorder() -> list[Any]:
        root = getattr(cache, "root_node", None)
        if root is None:
            return []
        visited: set[Any] = set()
        stack: list[Any] = [root]
        order: list[Any] = []
        while stack:
            node = stack.pop()
            if node is None or node in visited:
                continue
            visited.add(node)
            order.append(node)

            children = getattr(node, "children", None)
            if isinstance(children, dict):
                items = list(children.items())
                items.sort(key=lambda kv: _stable_child_key(kv[0]))
                for _k, child in reversed(items):
                    stack.append(child)
            elif children is not None:
                try:
                    child_list = list(children)
                except Exception:
                    child_list = []
                # Keep best-effort stability for non-dict children.
                child_list.sort(key=lambda c: (type(c).__name__, repr(c)))
                for child in reversed(child_list):
                    stack.append(child)
        order.reverse()
        return order

    # Cache node metadata computed from block IDs.
    # Value: (num_blocks, page_counts, sort_key, tie64)
    node_cache: dict[Any, tuple[int, dict[int, int], tuple[int, ...], int]] = {}

    persistent_cache = getattr(cache, "_kvcached_shrink_node_info_cache", None)
    if persistent_cache is None:
        persistent_cache = weakref.WeakKeyDictionary()
        setattr(cache, "_kvcached_shrink_node_info_cache", persistent_cache)

    def _tensor_sig(t: torch.Tensor) -> tuple[int, int, str, int, torch.dtype]:
        dev = t.device
        return (
            int(t.data_ptr()),
            int(t.numel()),
            str(dev.type),
            int(dev.index if dev.index is not None else -1),
            t.dtype,
        )

    def _get_node_info(
        node: Any,
    ) -> tuple[int, dict[int, int], tuple[int, ...], int] | None:
        cached = node_cache.get(node)
        if cached is not None:
            return cached

        val = getattr(node, "value", None)
        if val is None:
            try:
                persistent_cache.pop(node, None)
            except Exception:
                pass
            return None

        if isinstance(val, torch.Tensor):
            try:
                sig = _tensor_sig(val)
            except Exception:
                sig = None
            if sig is not None:
                cached2 = persistent_cache.get(node)
                if cached2 is not None:
                    cached_sig, info = cached2
                    if cached_sig == sig:
                        node_cache[node] = info
                        return info
        else:
            sig = None
        try:
            ids = val.detach().cpu().tolist()
        except Exception:
            try:
                ids = list(val)
            except Exception:
                return None
        if not ids:
            return None

        try:
            num_blocks = int(len(ids))
            first_id = int(ids[0])
            last_id = int(ids[-1])
        except Exception:
            return None

        priority = int(getattr(node, "priority", 0) or 0)

        min_id = first_id
        max_id = first_id
        sum32 = 0
        xor32 = 0

        # Deterministic 64-bit hash to avoid heap tie-breaking on node.__lt__.
        h64 = 1469598103934665603  # FNV-1a offset basis

        counts: dict[int, int] = defaultdict(int)
        for x in ids:
            try:
                i = int(x)
            except Exception:
                return None
            if i < min_id:
                min_id = i
            if i > max_id:
                max_id = i
            sum32 = (sum32 + i) & 0xFFFFFFFF
            xor32 = (xor32 ^ i) & 0xFFFFFFFF
            h64 = (h64 ^ (i & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
            h64 = (h64 * 1099511628211) & 0xFFFFFFFFFFFFFFFF

            pid = page_allocator.get_page_id(i, block_mem_size)
            counts[int(pid)] += 1

        sort_key = (
            priority,  # lower evicted first
            first_id,  # stable across TP ranks
            num_blocks,  # smaller nodes first
            last_id,  # stable across TP ranks
            min_id,
            max_id,
            sum32,
            xor32,
        )

        info = (num_blocks, dict(counts), sort_key, int(h64))
        node_cache[node] = info
        if sig is not None:
            try:
                persistent_cache[node] = (sig, info)
            except Exception:
                pass
        return info

    def _node_token_len(node: Any) -> int:
        # Prefer key length to avoid forcing device->host copies of node.value.
        try:
            key = getattr(node, "key", None)
            if key is not None:
                return int(len(key))
        except Exception:
            pass
        info = _get_node_info(node)
        if info is None:
            return 0
        num_blocks, _counts, _sort_key, _tie64 = info
        return int(num_blocks)

    total_freed_pages = 0
    for round_idx in range(max_rounds):
        round_start = time.perf_counter()
        if _time_exceeded():
            break

        before_inuse = int(page_allocator.get_num_inuse_pages())
        if before_inuse <= target_num_pages:
            break

        pages_needed = before_inuse - int(target_num_pages)
        t0 = time.perf_counter()
        allocated_by_page = _get_allocated_by_page()
        t_alloc_ms = (time.perf_counter() - t0) * 1000
        if not allocated_by_page:
            break

        t0 = time.perf_counter()
        postorder_nodes = _iter_nodes_postorder()
        t_post_ms = (time.perf_counter() - t0) * 1000
        if not postorder_nodes:
            break

        # Compute subtree token count and "contains locked node" flags.
        t0 = time.perf_counter()
        subtree_tokens: dict[Any, int] = {}
        subtree_has_lock: dict[Any, bool] = {}
        for node in postorder_nodes:
            total = _node_token_len(node)
            has_lock = int(getattr(node, "lock_ref", 0) or 0) > 0

            children = getattr(node, "children", None)
            if isinstance(children, dict):
                child_nodes = children.values()
            elif children is not None:
                try:
                    child_nodes = list(children)
                except Exception:
                    child_nodes = []
            else:
                child_nodes = []

            for child in child_nodes:
                total += int(subtree_tokens.get(child, 0))
                if subtree_has_lock.get(child, False):
                    has_lock = True

            subtree_tokens[node] = int(total)
            subtree_has_lock[node] = bool(has_lock)
            if _time_exceeded():
                break
        t_subtree_ms = (time.perf_counter() - t0) * 1000
        if _time_exceeded():
            break

        # Compute per-page coverage and locked pages.
        t0 = time.perf_counter()
        locked_pages: set[int] = set()
        known_blocks_by_page: dict[int, int] = defaultdict(int)
        nodes_by_page: dict[int, list[Any]] = defaultdict(list)

        for node in postorder_nodes:
            info = _get_node_info(node)
            if info is None:
                continue
            _num_blocks, counts, _sort_key, _tie64 = info
            if not counts:
                continue

            lock_ref = int(getattr(node, "lock_ref", 0) or 0)
            for pid, cnt in counts.items():
                if pid not in allocated_by_page:
                    continue
                known_blocks_by_page[int(pid)] += int(cnt)
                if lock_ref > 0:
                    locked_pages.add(int(pid))
                else:
                    nodes_by_page[int(pid)].append(node)
            if _time_exceeded():
                break
        t_page_scan_ms = (time.perf_counter() - t0) * 1000
        if _time_exceeded():
            break

        # Exclude pages that contain pinned blocks (e.g., reserved null block).
        excluded_pages: set[int] = set()
        null_block = getattr(allocator, "null_block", None)
        if null_block:
            try:
                null_pid = int(page_allocator.get_page_id(int(null_block[0]), block_mem_size))
                excluded_pages.add(null_pid)
            except Exception:
                pass

        t0 = time.perf_counter()
        candidate_pages: list[tuple[int, int, int, list[Any]]] = []
        for pid, allocated in allocated_by_page.items():
            if pid in excluded_pages or pid in locked_pages:
                continue
            if known_blocks_by_page.get(pid, 0) < int(allocated):
                continue

            page_nodes = nodes_by_page.get(pid)
            if not page_nodes:
                continue

            page_nodes_set = set(page_nodes)
            roots: list[Any] = []
            for node in page_nodes_set:
                p = getattr(node, "parent", None)
                while p is not None and p != getattr(cache, "root_node", None):
                    if p in page_nodes_set:
                        break
                    p = getattr(p, "parent", None)
                else:
                    roots.append(node)

            if not roots:
                continue
            if any(subtree_has_lock.get(r, False) for r in roots):
                continue

            roots.sort(
                key=lambda n: (
                    (_get_node_info(n) or (0, {}, (), 0))[2],
                    (_get_node_info(n) or (0, {}, (), 0))[3],
                )
            )

            cost_tokens = 0
            for r in roots:
                cost_tokens += int(subtree_tokens.get(r, 0))

            candidate_pages.append((int(cost_tokens), int(allocated), int(pid), roots))
            if _time_exceeded():
                break
        t_candidate_ms = (time.perf_counter() - t0) * 1000
        if _time_exceeded():
            break

        if not candidate_pages:
            break

        t0 = time.perf_counter()
        candidate_pages.sort(key=lambda t: (t[0], t[1], t[2]))
        if len(candidate_pages) >= pages_needed:
            selected = candidate_pages[:pages_needed]
        else:
            selected = candidate_pages

        selected_pages = [t[2] for t in selected]
        selected_pages_set = set(selected_pages)
        remaining: dict[int, int] = {pid: allocated_by_page[pid] for pid in selected_pages}

        # Collect all selected roots and de-duplicate ancestor/descendant overlaps.
        selected_roots: list[Any] = []
        for _cost, _allocated, _pid, roots in selected:
            selected_roots.extend(roots)
        selected_roots_set = set(selected_roots)

        def _depth(node: Any) -> int:
            d = 0
            p = node
            root = getattr(cache, "root_node", None)
            while p is not None and p != root:
                d += 1
                p = getattr(p, "parent", None)
            return d

        roots_sorted = sorted(
            selected_roots_set,
            key=lambda n: (
                _depth(n),
                (_get_node_info(n) or (0, {}, (), 0))[2],
                (_get_node_info(n) or (0, {}, (), 0))[3],
            ),
        )
        minimal_set: set[Any] = set()
        for r in roots_sorted:
            p = getattr(r, "parent", None)
            while p is not None and p != getattr(cache, "root_node", None):
                if p in minimal_set:
                    break
                p = getattr(p, "parent", None)
            else:
                minimal_set.add(r)

        # Traverse selected subtrees only (avoid scanning the whole tree again).
        in_target: set[Any] = set()
        leaves: list[Any] = []
        stack2: list[Any] = list(minimal_set)
        while stack2:
            node = stack2.pop()
            if node is None or node in in_target:
                continue
            in_target.add(node)

            children = getattr(node, "children", None)
            if isinstance(children, dict):
                items = list(children.items())
                items.sort(key=lambda kv: _stable_child_key(kv[0]))
                child_list = [c for _k, c in items]
            elif children is not None:
                try:
                    child_list = list(children)
                except Exception:
                    child_list = []
                child_list.sort(key=lambda c: (type(c).__name__, repr(c)))
            else:
                child_list = []

            if not child_list:
                leaves.append(node)
                continue

            for child in reversed(child_list):
                stack2.append(child)
        pushed: set[Any] = set()
        candidates: list[tuple[tuple[int, ...], int, Any]] = []

        def _push(node: Any) -> None:
            if node in pushed:
                return
            if node not in in_target:
                return
            info = _get_node_info(node)
            if info is None:
                return
            _num_blocks, _counts, sort_key, tie64 = info
            heapq.heappush(candidates, (sort_key, tie64, node))
            pushed.add(node)

        for node in leaves:
            _push(node)

        t_select_ms = (time.perf_counter() - t0) * 1000

        num_nodes_evicted = 0
        pending_free: list[torch.Tensor] = []
        t0 = time.perf_counter()
        while remaining and candidates and not _time_exceeded():
            _sort_key, _tie64, node = heapq.heappop(candidates)

            if node not in in_target:
                continue
            if int(getattr(node, "lock_ref", 1) or 0) != 0:
                continue
            if len(getattr(node, "children", {})) != 0:
                continue

            parent = getattr(node, "parent", None)
            if parent is None:
                continue
            child_key = cache.get_child_key_fn(node.key)
            if getattr(parent, "children", {}).get(child_key) is not node:
                continue

            info = _get_node_info(node)
            if info is None:
                continue
            _num_blocks, counts, _sort_key2, _tie64b = info

            # Batch frees to reduce TP-wide unmap calls (the expensive part).
            try:
                val = getattr(node, "value", None)
                if isinstance(val, torch.Tensor):
                    if val.numel() > 0:
                        pending_free.append(val)
                elif val is not None:
                    try:
                        ids = [int(x) for x in val]
                    except Exception:
                        ids = []
                    if ids:
                        allocator.free(ids)
            except Exception:
                # Best-effort: fall back to immediate free.
                try:
                    val = getattr(node, "value", None)
                    if isinstance(val, torch.Tensor):
                        if val.numel() > 0:
                            pending_free.append(val)
                    elif val is not None:
                        ids = [int(x) for x in val]
                        if ids:
                            allocator.free(ids)
                except Exception:
                    pass
            cache._delete_leaf(node)
            cache._record_remove_event(node)
            num_nodes_evicted += 1

            for pid, cnt in counts.items():
                if pid in remaining:
                    remaining[pid] -= int(cnt)
                    if remaining[pid] <= 0:
                        del remaining[pid]

            p = parent
            while (
                p is not None
                and p != getattr(cache, "root_node", None)
                and p in in_target
                and int(getattr(p, "lock_ref", 1) or 0) == 0
                and len(getattr(p, "children", {})) == 0
            ):
                _push(p)
                p = getattr(p, "parent", None)

        t_evict_ms = (time.perf_counter() - t0) * 1000

        free_ms = 0.0
        if pending_free:
            free_start = time.perf_counter()
            if len(pending_free) == 1:
                cache.token_to_kv_pool_allocator.free(pending_free[0])
            else:
                cache.token_to_kv_pool_allocator.free(torch.cat(pending_free))
            free_ms = (time.perf_counter() - free_start) * 1000

        after_inuse = int(page_allocator.get_num_inuse_pages())
        freed = before_inuse - after_inuse
        if rank == 0:
            logger.debug(
                f"[shrink-evict] round={round_idx} before_inuse={before_inuse} "
                f"after_inuse={after_inuse} target={target_num_pages} "
                f"selected_pages={sorted(selected_pages_set)} freed_pages={freed} "
                f"evicted_nodes={num_nodes_evicted} free_ms={free_ms:.1f} "
                f"round_ms={(time.perf_counter() - round_start) * 1000:.1f} "
                f"alloc_ms={t_alloc_ms:.1f} post_ms={t_post_ms:.1f} "
                f"subtree_ms={t_subtree_ms:.1f} scan_ms={t_page_scan_ms:.1f} "
                f"cand_ms={t_candidate_ms:.1f} select_ms={t_select_ms:.1f} "
                f"evict_ms={t_evict_ms:.1f} nodes={len(postorder_nodes)} "
                f"candidates={len(candidate_pages)}"
            )

        if freed <= 0 and num_nodes_evicted == 0:
            break
        total_freed_pages += int(freed)

    return int(total_freed_pages)

class RadixCacheShrinkEvictionPatch(VersionAwarePatch, BasePatch):
    """Add page-aware eviction to help kvcached shrink succeed.

    This patch:
      - Registers RadixCache instances to the underlying KVCacheManager.
      - Injects a deterministic, page-aware eviction method that preferentially
        frees whole physical pages (reducing internal fragmentation).
    """

    library = "sglang"
    target_module = "sglang.srt.mem_cache.radix_cache"
    target_class = "RadixCache"
    patch_name = "radix_cache_shrink_eviction"

    def apply(self, radix_mod: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False
        return self.patch_radix_cache(radix_mod)

    @version_range(SGLANG_ALL_RANGE)
    def patch_radix_cache(self, radix_mod: types.ModuleType) -> bool:
        RadixCache = self._get_target_class(radix_mod)
        if RadixCache is None:
            return False

        if self._is_already_patched(RadixCache, "__kvcached_shrink_eviction__"):
            return True

        original_init = getattr(RadixCache, "__init__", None)
        if original_init is None:
            return False

        def _register_with_allocator(cache) -> None:
            allocator = getattr(getattr(cache, "token_to_kv_pool_allocator", None),
                                "kvcached_allocator", None)
            if allocator is None:
                return
            ws = getattr(allocator, "_kvcached_radix_caches", None)
            if ws is None:
                ws = weakref.WeakSet()
                setattr(allocator, "_kvcached_radix_caches", ws)
            ws.add(cache)

        def _kvcached_evict_pages_for_shrink(cache, target_num_pages: int) -> int:
            return _kvcached_evict_pages_for_shrink_impl(cache, target_num_pages)

        def _wrapped_init(self, *args: Any, **kwargs: Any):
            original_init(self, *args, **kwargs)
            try:
                _register_with_allocator(self)
            except Exception:
                pass

        RadixCache.__init__ = _wrapped_init  # type: ignore[assignment]
        RadixCache._kvcached_evict_pages_for_shrink = _kvcached_evict_pages_for_shrink  # type: ignore[attr-defined]

        # Batch radix eviction frees when kvcached is enabled to reduce the number
        # of TP-wide unmap calls. This keeps eviction semantics the same, but
        # performs the underlying allocator free in a single batched call.
        original_evict = getattr(RadixCache, "evict", None)
        if original_evict is not None and not self._is_already_patched(original_evict):

            def _wrapped_evict(self, num_tokens: int):
                if not enable_kvcached() or getattr(self, "disable", False):
                    return original_evict(self, num_tokens)

                tok_alloc = getattr(self, "token_to_kv_pool_allocator", None)
                if tok_alloc is None or not hasattr(tok_alloc, "kvcached_allocator"):
                    return original_evict(self, num_tokens)

                start_time = time.perf_counter()
                leaves = self._collect_leaves()
                eviction_heap = [
                    (self.eviction_strategy.get_priority(node), node) for node in leaves
                ]
                heapq.heapify(eviction_heap)

                num_evicted = 0
                pending_free: list[torch.Tensor] = []
                while num_evicted < num_tokens and eviction_heap:
                    _priority, x = heapq.heappop(eviction_heap)

                    val = getattr(x, "value", None)
                    if isinstance(val, torch.Tensor) and val.numel() > 0:
                        pending_free.append(val.to(dtype=torch.int64, copy=False))
                        num_evicted += int(val.numel())
                    else:
                        # Fallback: keep the original behavior.
                        tok_alloc.free(x.value)
                        num_evicted += len(x.value)

                    self._delete_leaf(x)

                    if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                        new_priority = self.eviction_strategy.get_priority(x.parent)
                        heapq.heappush(eviction_heap, (new_priority, x.parent))

                    self._record_remove_event(x)

                if pending_free:
                    if len(pending_free) == 1:
                        tok_alloc.free(pending_free[0])
                    else:
                        tok_alloc.free(torch.cat(pending_free))

                self.update_eviction_metrics(num_evicted, start_time)

            self._mark_as_patched(_wrapped_evict)
            setattr(RadixCache, "evict", _wrapped_evict)

        self._mark_as_patched(RadixCache, "__kvcached_shrink_eviction__")
        return True
