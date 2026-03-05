"""
HiRadix cache-specific SGLang patches.

This patch adds a kvcached shrink helper for HiRadixCache. Compared to the
RadixCache shrink eviction, HiRadixCache supports hierarchical cache with an
optional host backup. For shrink, we prefer device-evict (keep host cache) when
possible, and fall back to subtree deletion when host backup is not possible.
"""

from __future__ import annotations

import heapq
import time
import weakref
from collections import defaultdict
from typing import Any

import torch

from kvcached.integration.patch_base import BasePatch
from kvcached.integration.sglang.patches import SGLANG_ALL_RANGE
from kvcached.integration.version_utils import VersionAwarePatch, version_range
from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


def _kvcached_evict_pages_for_shrink_impl(cache, target_num_pages: int) -> int:
    """Evict HiRadixCache entries (not in compute) to help kvcached shrink succeed.

    Goal: free whole kvcached physical pages to satisfy shrink.

    Strategy (page-aware + lock-aware + host-friendly):
      1) Identify "fully releasable" pages: pages fully covered by device-resident
         radix nodes and without any blocks belonging to locked nodes (lock_ref > 0).
      2) If there are more fully releasable pages than needed, choose the subset
         with the lowest eviction cost (estimated by subtree token count).
         Otherwise, evict all fully releasable pages.
      3) Evict nodes in the selected subtrees in a deterministic device-postorder:
           - If a node is already backuped (host_value exists), evict device blocks
             and keep the node (value=None) so host cache remains usable.
           - Otherwise, try to back up device blocks to host first, then evict.
           - If host backup is not possible, fall back to deleting the node (and
             dropping any host-only descendants) to prioritize shrink success.

    Notes:
      - Best-effort and time-bounded.
      - Determinism matters for TP: all ordering uses block-id-derived keys.
      - Never leave a node as evicted-but-not-backuped in the tree (it breaks
        HiRadixCache.match_prefix assumptions).
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

    tok_alloc = getattr(cache, "token_to_kv_pool_allocator", None)
    allocator = getattr(tok_alloc, "kvcached_allocator", None)
    if allocator is None:
        return 0

    page_allocator = allocator.page_allocator
    if getattr(cache, "page_size", 1) != 1:
        # The kvcached SGLang integration is only validated with page_size=1.
        return 0

    block_mem_size = int(allocator.block_mem_size)
    rank = _safe_rank()

    # Time budget and multi-round loop are defensive: freeing may be partial due
    # to internal fragmentation and/or eviction constraints.
    max_rounds = 5
    max_time_ms = 1500.0
    start_time = time.perf_counter()

    def _time_exceeded() -> bool:
        return (time.perf_counter() - start_time) * 1000 >= max_time_ms

    def _stable_child_key(x: Any) -> tuple[str, str]:
        # Keys may be custom objects; keep sort stable without relying on __lt__.
        return (type(x).__name__, repr(x))

    def _is_device_leaf(node: Any) -> bool:
        if getattr(node, "value", None) is None:
            return False
        if node == getattr(cache, "root_node", None):
            return False
        children = getattr(node, "children", None)
        if not children:
            return True
        try:
            child_values = children.values() if isinstance(children, dict) else list(children)
        except Exception:
            child_values = []
        for c in child_values:
            if getattr(c, "value", None) is not None:
                return False
        return True

    def _get_allocated_by_page() -> dict[int, int]:
        allocated_by_page: dict[int, int] = {}
        for page in list(getattr(allocator, "full_pages", {}).values()) + list(
            getattr(allocator, "avail_pages", {}).values()
        ):
            num_kv_blocks = getattr(page, "num_kv_blocks", None)
            if num_kv_blocks is None:
                continue
            allocated = int(num_kv_blocks) - int(page.num_free_blocks())
            if allocated > 0:
                allocated_by_page[int(page.page_id)] = int(allocated)
        return allocated_by_page

    def _iter_nodes_postorder_device() -> list[Any]:
        """Postorder traversal over device-resident nodes only (value is not None)."""
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
                    if getattr(child, "value", None) is not None:
                        stack.append(child)
            elif children is not None:
                try:
                    child_list = list(children)
                except Exception:
                    child_list = []
                child_list.sort(key=lambda c: (type(c).__name__, repr(c)))
                for child in reversed(child_list):
                    if getattr(child, "value", None) is not None:
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

    def _try_backup_to_host(node: Any, prefer_evict_host: bool) -> bool:
        """Best-effort device->host backup for one node.

        Returns True if node.host_value is set (backuped), otherwise False.
        """
        if getattr(node, "host_value", None) is not None:
            return True
        val = getattr(node, "value", None)
        if val is None:
            return False

        cc = getattr(cache, "cache_controller", None)
        if cc is None:
            return False

        # Use a negative tag to mark shrink-induced writes. This avoids
        # interfering with ongoing_write_through and lock_ref management.
        try:
            tag_id = -(int(getattr(node, "id")) + 1)
        except Exception:
            tag_id = -1

        host_indices = cc.write(device_indices=val, node_id=int(tag_id))
        if host_indices is None and prefer_evict_host:
            # Host memory is per-rank. This eviction must be deterministic; we
            # implement a best-effort deterministic eviction below.
            _evict_host_deterministic(int(len(val)))
            host_indices = cc.write(device_indices=val, node_id=int(tag_id))

        if host_indices is None:
            return False

        # Wait for the corresponding ack to ensure host KV is ready before
        # freeing device blocks. If we can't find a matching ack, treat it as
        # failure and free the allocated host indices to avoid leaks.
        found = False
        try:
            for _start, finish_event, ack_list in reversed(cc.ack_write_queue):
                try:
                    if int(tag_id) in ack_list:
                        finish_event.synchronize()
                        found = True
                        break
                except Exception:
                    continue
        except Exception:
            found = False

        if not found:
            try:
                cc.mem_pool_host.free(host_indices)
            except Exception:
                pass
            return False

        try:
            node.host_value = host_indices
        except Exception:
            try:
                cc.mem_pool_host.free(host_indices)
            except Exception:
                pass
            return False
        return True

    def _evict_host_deterministic(num_tokens: int) -> int:
        """Best-effort deterministic host eviction to free host slots.

        This is only used to make space for host backup during shrink. It avoids
        time-based priorities and uses a stable ordering derived from node ids.
        """
        if num_tokens <= 0:
            return 0
        cc = getattr(cache, "cache_controller", None)
        if cc is None:
            return 0

        root = getattr(cache, "root_node", None)
        if root is None:
            return 0

        def _host_leaf(node: Any) -> bool:
            if getattr(node, "value", None) is not None:
                return False
            if node == root:
                return False
            if getattr(node, "host_value", None) is None:
                return False
            if int(getattr(node, "host_ref_counter", 0) or 0) > 0:
                return False
            children = getattr(node, "children", None)
            try:
                if children and len(children) > 0:
                    return False
            except Exception:
                # If we can't determine, be conservative and treat as non-leaf.
                return False
            return True

        # Collect host-only leaves and build a deterministic heap.
        heap: list[tuple[int, int, int, Any]] = []
        stack: list[Any] = list(getattr(root, "children", {}).values())
        while stack:
            n = stack.pop()
            if _host_leaf(n):
                try:
                    tok = int(len(getattr(n, "host_value")))
                except Exception:
                    tok = 0
                try:
                    nid = int(getattr(n, "id"))
                except Exception:
                    nid = 0
                # Sort: smaller token loss first, then stable id.
                heapq.heappush(heap, (tok, nid, tok, n))
                continue

            children = getattr(n, "children", None)
            if isinstance(children, dict):
                items = list(children.items())
                items.sort(key=lambda kv: _stable_child_key(kv[0]))
                stack.extend([c for _k, c in reversed(items)])
            elif children is not None:
                try:
                    child_list = list(children)
                except Exception:
                    child_list = []
                child_list.sort(key=lambda c: (type(c).__name__, repr(c)))
                stack.extend(reversed(child_list))

        freed = 0
        while heap and freed < num_tokens and not _time_exceeded():
            _tok, _nid, tok, node = heapq.heappop(heap)
            hv = getattr(node, "host_value", None)
            if hv is None:
                continue
            if int(getattr(node, "host_ref_counter", 0) or 0) > 0:
                continue

            try:
                cc.evict_host(hv)
            except Exception:
                try:
                    cc.mem_pool_host.free(hv)
                except Exception:
                    continue

            try:
                node.host_value = None
            except Exception:
                pass

            # Remove from tree.
            try:
                child_key = cache.get_child_key_fn(node.key)
                parent = getattr(node, "parent", None)
                if parent is not None:
                    parent.children.pop(child_key, None)
            except Exception:
                pass

            freed += int(tok)

            # Parent might become a host leaf if it is evicted and has no children.
            parent = getattr(node, "parent", None)
            while parent is not None and parent != root:
                if getattr(parent, "value", None) is not None:
                    break
                if getattr(parent, "host_value", None) is None:
                    break
                if int(getattr(parent, "host_ref_counter", 0) or 0) > 0:
                    break
                try:
                    if len(getattr(parent, "children", {})) != 0:
                        break
                except Exception:
                    break
                try:
                    ptok = int(len(getattr(parent, "host_value")))
                except Exception:
                    ptok = 0
                try:
                    pid = int(getattr(parent, "id"))
                except Exception:
                    pid = 0
                heapq.heappush(heap, (ptok, pid, ptok, parent))
                parent = getattr(parent, "parent", None)

        return int(freed)

    def _subtree_has_protected_host(node: Any) -> bool:
        """Return True if subtree contains host nodes protected from eviction."""
        stack: list[Any] = [node]
        while stack:
            n = stack.pop()
            if int(getattr(n, "host_ref_counter", 0) or 0) > 0:
                return True
            children = getattr(n, "children", None)
            if isinstance(children, dict):
                stack.extend(children.values())
            elif children is not None:
                try:
                    stack.extend(list(children))
                except Exception:
                    pass
        return False

    def _free_host_subtree(node: Any) -> None:
        """Free host_value for all evicted descendants (best-effort)."""
        cc = getattr(cache, "cache_controller", None)
        if cc is None:
            return

        host_tensors: list[torch.Tensor] = []
        stack: list[Any] = list(getattr(node, "children", {}).values())
        while stack:
            n = stack.pop()
            hv = getattr(n, "host_value", None)
            if hv is not None:
                if int(getattr(n, "host_ref_counter", 0) or 0) == 0:
                    host_tensors.append(hv)
                try:
                    n.host_value = None
                except Exception:
                    pass
            children = getattr(n, "children", None)
            if isinstance(children, dict):
                stack.extend(children.values())
            elif children is not None:
                try:
                    stack.extend(list(children))
                except Exception:
                    pass

        if host_tensors:
            try:
                cc.mem_pool_host.free(torch.cat(host_tensors))
            except Exception:
                try:
                    for t in host_tensors:
                        cc.mem_pool_host.free(t)
                except Exception:
                    pass

    total_freed_pages = 0
    for round_idx in range(max_rounds):
        round_start = time.perf_counter()
        if _time_exceeded():
            break

        before_inuse = int(page_allocator.get_num_inuse_pages())
        if before_inuse <= int(target_num_pages):
            break

        pages_needed = before_inuse - int(target_num_pages)
        allocated_by_page = _get_allocated_by_page()
        if not allocated_by_page:
            break

        postorder_nodes = _iter_nodes_postorder_device()
        if not postorder_nodes:
            break

        # Compute subtree token count and "contains locked node" flags.
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
                # Only device-resident children participate in device shrink.
                if getattr(child, "value", None) is None:
                    continue
                total += int(subtree_tokens.get(child, 0))
                if subtree_has_lock.get(child, False):
                    has_lock = True

            subtree_tokens[node] = int(total)
            subtree_has_lock[node] = bool(has_lock)
            if _time_exceeded():
                break
        if _time_exceeded():
            break

        # Compute per-page coverage and locked pages.
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
        if _time_exceeded():
            break

        if not candidate_pages:
            break

        candidate_pages.sort(key=lambda t: (t[0], t[1], t[2]))
        selected = candidate_pages[:pages_needed] if len(candidate_pages) >= pages_needed else candidate_pages
        selected_pages = [t[2] for t in selected]
        selected_pages_set = set(selected_pages)
        remaining: dict[int, int] = {pid: int(allocated_by_page[pid]) for pid in selected_pages}

        # De-duplicate ancestor/descendant overlaps.
        selected_roots_set: set[Any] = set()
        for _cost, _allocated, _pid, roots in selected:
            selected_roots_set.update(roots)

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

        # Traverse selected subtrees (device nodes only) and collect initial device-leaves.
        in_target: set[Any] = set()
        device_leaves: list[Any] = []
        stack2: list[Any] = list(minimal_set)
        while stack2:
            node = stack2.pop()
            if node is None or node in in_target:
                continue
            if getattr(node, "value", None) is None:
                continue
            in_target.add(node)

            children = getattr(node, "children", None)
            child_list: list[Any] = []
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

            has_device_child = False
            for child in child_list:
                if getattr(child, "value", None) is not None:
                    has_device_child = True
                    stack2.append(child)

            if not has_device_child and _is_device_leaf(node):
                device_leaves.append(node)

        pushed: set[Any] = set()
        candidates: list[tuple[tuple[int, ...], int, Any]] = []

        def _push(node: Any) -> None:
            if node in pushed:
                return
            if node not in in_target:
                return
            if int(getattr(node, "lock_ref", 1) or 0) != 0:
                return
            if not _is_device_leaf(node):
                return
            info = _get_node_info(node)
            if info is None:
                return
            _num_blocks, _counts, sort_key, tie64 = info
            heapq.heappush(candidates, (sort_key, tie64, node))
            pushed.add(node)

        for node in device_leaves:
            _push(node)

        num_nodes_evicted = 0
        pending_free: list[torch.Tensor] = []

        # Prefer eviction patterns that keep host cache. Host backup is done
        # synchronously per-node (only for nodes we actually evict).
        while remaining and candidates and not _time_exceeded():
            _sort_key, _tie64, node = heapq.heappop(candidates)

            if node not in in_target:
                continue
            if int(getattr(node, "lock_ref", 1) or 0) != 0:
                continue
            if not _is_device_leaf(node):
                continue

            parent = getattr(node, "parent", None)
            if parent is None:
                continue

            info = _get_node_info(node)
            if info is None:
                continue
            _num_blocks, counts, _sort_key2, _tie64b = info

            val = getattr(node, "value", None)
            if val is None:
                continue

            # 1) Keep host cache when possible.
            if getattr(node, "host_value", None) is not None:
                pending_free.append(val)
                try:
                    cache.evictable_size_ -= int(len(val))
                except Exception:
                    pass
                node.value = None
                # Keep internal eviction sets consistent with SGLang's
                # HiRadixCache._evict_backuped semantics.
                try:
                    cache._update_leaf_status(node)
                except Exception:
                    pass
                try:
                    cache._update_host_leaf_status(node)
                except Exception:
                    pass
                try:
                    cache._update_leaf_status(parent)
                except Exception:
                    pass
            else:
                # 2) Try host backup first. If host is full, optionally evict
                #    some host-only cache deterministically.
                backed = _try_backup_to_host(node, prefer_evict_host=True)
                if backed:
                    pending_free.append(val)
                    try:
                        cache.evictable_size_ -= int(len(val))
                    except Exception:
                        pass
                    node.value = None
                    try:
                        cache._update_leaf_status(node)
                    except Exception:
                        pass
                    try:
                        cache._update_host_leaf_status(node)
                    except Exception:
                        pass
                    try:
                        cache._update_leaf_status(parent)
                    except Exception:
                        pass
                else:
                    # 3) Shrink success first: drop subtree.
                    # Avoid deleting if there are protected host nodes in the subtree.
                    if _subtree_has_protected_host(node):
                        continue
                    _free_host_subtree(node)
                    pending_free.append(val)
                    try:
                        cache._delete_leaf(node)
                        cache._record_remove_event(node)
                    except Exception:
                        # Best-effort.
                        try:
                            cache._delete_leaf(node)
                        except Exception:
                            pass

            num_nodes_evicted += 1

            for pid, cnt in counts.items():
                if pid in remaining:
                    remaining[pid] -= int(cnt)
                    if remaining[pid] <= 0:
                        del remaining[pid]

            # Push parents that become device-leaves.
            p = parent
            while (
                p is not None
                and p != getattr(cache, "root_node", None)
                and p in in_target
                and int(getattr(p, "lock_ref", 1) or 0) == 0
                and _is_device_leaf(p)
            ):
                _push(p)
                p = getattr(p, "parent", None)

        free_ms = 0.0
        if pending_free:
            free_start = time.perf_counter()
            try:
                if len(pending_free) == 1:
                    tok_alloc.free(pending_free[0])
                else:
                    tok_alloc.free(torch.cat(pending_free))
            except Exception:
                # Best-effort: fall back to per-node frees.
                try:
                    for t in pending_free:
                        tok_alloc.free(t)
                except Exception:
                    pass
            free_ms = (time.perf_counter() - free_start) * 1000

        after_inuse = int(page_allocator.get_num_inuse_pages())
        freed = before_inuse - after_inuse
        if rank == 0:
            logger.debug(
                f"[hi-shrink-evict] round={round_idx} before_inuse={before_inuse} "
                f"after_inuse={after_inuse} target={target_num_pages} "
                f"selected_pages={sorted(selected_pages_set)} freed_pages={freed} "
                f"evicted_nodes={num_nodes_evicted} free_ms={free_ms:.1f} "
                f"round_ms={(time.perf_counter() - round_start) * 1000:.1f}"
            )

        if freed <= 0 and num_nodes_evicted == 0:
            break
        total_freed_pages += int(freed)

    return int(total_freed_pages)


class HiRadixCacheShrinkEvictionPatch(VersionAwarePatch, BasePatch):
    """Add page-aware eviction to HiRadixCache to help kvcached shrink succeed."""

    library = "sglang"
    target_module = "sglang.srt.mem_cache.hiradix_cache"
    target_class = "HiRadixCache"
    patch_name = "hiradix_cache_shrink_eviction"

    def apply(self, hiradix_mod) -> bool:
        if not self.initialize_version_info():
            return False
        return self.patch_hiradix_cache(hiradix_mod)

    @version_range(SGLANG_ALL_RANGE)
    def patch_hiradix_cache(self, hiradix_mod) -> bool:
        HiRadixCache = self._get_target_class(hiradix_mod)
        if HiRadixCache is None:
            return False

        if self._is_already_patched(HiRadixCache, "__kvcached_hiradix_shrink_eviction__"):
            return True

        original_init = getattr(HiRadixCache, "__init__", None)
        if original_init is None:
            return False

        def _register_with_allocator(cache) -> None:
            allocator = getattr(getattr(cache, "token_to_kv_pool_allocator", None), "kvcached_allocator", None)
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

        HiRadixCache.__init__ = _wrapped_init  # type: ignore[assignment]
        HiRadixCache._kvcached_evict_pages_for_shrink = _kvcached_evict_pages_for_shrink  # type: ignore[attr-defined]

        # Patch writing_check to tolerate shrink-induced write acks. Shrink uses
        # negative node_ids to avoid touching ongoing_write_through.
        original_writing_check = getattr(HiRadixCache, "writing_check", None)
        if original_writing_check is not None and not self._is_already_patched(original_writing_check):

            def _wrapped_writing_check(self, write_back: bool = False):
                try:
                    cc = getattr(self, "cache_controller", None)
                    if cc is None:
                        return original_writing_check(self, write_back=write_back)

                    # Drain finished shrink-only acks even when there is no
                    # ongoing_write_through, to avoid ack queue growth.
                    if not getattr(self, "ongoing_write_through", None):
                        drained = 0
                        for _start, finish_event, ack_list in list(getattr(cc, "ack_write_queue", [])):
                            try:
                                if not finish_event.query():
                                    break
                            except Exception:
                                break
                            # Only drain acks that are shrink-only (negative ids).
                            try:
                                if any(int(x) >= 0 for x in ack_list):
                                    break
                            except Exception:
                                break
                            drained += 1
                        if drained > 0:
                            del cc.ack_write_queue[:drained]
                        return

                    # Normal path: run original logic but ignore unknown ack ids.
                    # This prevents KeyError for shrink-only negative ids.
                    if write_back:
                        while len(self.ongoing_write_through) > 0:
                            for _, finish_event, ack_list in cc.ack_write_queue:
                                finish_event.synchronize()
                                for ack_id in ack_list:
                                    self.ongoing_write_through.pop(ack_id, None)
                            cc.ack_write_queue.clear()
                            assert len(self.ongoing_write_through) == 0
                        return

                    finish_count = 0
                    for _, finish_event, _ack_list in cc.ack_write_queue:
                        if not finish_event.query():
                            break
                        finish_count += 1
                    queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
                    if getattr(self, "tp_world_size", 1) > 1:
                        torch.distributed.all_reduce(
                            queue_size,
                            op=torch.distributed.ReduceOp.MIN,
                            group=getattr(self, "tp_group", None),
                        )
                    finish_count = int(queue_size.item())
                    while finish_count > 0:
                        _, finish_event, ack_list = cc.ack_write_queue.pop(0)
                        finish_event.synchronize()
                        for ack_id in ack_list:
                            node = self.ongoing_write_through.pop(ack_id, None)
                            if node is None:
                                continue
                            self.dec_lock_ref(node)
                            if getattr(self, "enable_storage", False):
                                self.write_backup_storage(node)
                        finish_count -= 1
                    return
                except Exception:
                    return original_writing_check(self, write_back=write_back)

            self._mark_as_patched(_wrapped_writing_check)
            setattr(HiRadixCache, "writing_check", _wrapped_writing_check)

        self._mark_as_patched(HiRadixCache, "__kvcached_hiradix_shrink_eviction__")
        return True
