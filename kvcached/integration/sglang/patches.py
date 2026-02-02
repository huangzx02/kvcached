# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
SGLang-specific patches using unified patch infrastructure.
"""

import inspect
import heapq
import time
import types
import weakref
from collections import defaultdict
from typing import Any, Union
import torch

from kvcached.integration.patch_base import BasePatch, enable_kvcached
from kvcached.integration.version_utils import VersionAwarePatch, version_range
from kvcached.utils import get_kvcached_logger

# Version ranges for SGLang support
SGLANG_ALL_RANGE = ">=0.4.9"  # All supported versions

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
    max_time_ms = 100.0
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
    # Value: (block_ids, page_counts, sort_key, tie64)
    node_cache: dict[Any, tuple[list[int], dict[int, int], tuple[int, ...], int]] = {}

    def _get_node_info(
        node: Any,
    ) -> tuple[list[int], dict[int, int], tuple[int, ...], int] | None:
        cached = node_cache.get(node)
        if cached is not None:
            return cached

        val = getattr(node, "value", None)
        if val is None:
            return None
        try:
            ids = val.detach().cpu().tolist()
        except Exception:
            try:
                ids = list(val)
            except Exception:
                return None
        if not ids:
            return None

        block_ids: list[int] = []
        for x in ids:
            try:
                block_ids.append(int(x))
            except Exception:
                return None

        priority = int(getattr(node, "priority", 0) or 0)
        first_id = int(block_ids[0])
        last_id = int(block_ids[-1])

        min_id = first_id
        max_id = first_id
        sum32 = 0
        xor32 = 0

        # Deterministic 64-bit hash to avoid heap tie-breaking on node.__lt__.
        h64 = 1469598103934665603  # FNV-1a offset basis

        counts: dict[int, int] = defaultdict(int)
        for i in block_ids:
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
            len(block_ids),  # smaller nodes first
            last_id,  # stable across TP ranks
            min_id,
            max_id,
            sum32,
            xor32,
        )

        info = (block_ids, dict(counts), sort_key, int(h64))
        node_cache[node] = info
        return info

    def _node_token_len(node: Any) -> int:
        info = _get_node_info(node)
        if info is None:
            return 0
        ids, _counts, _sort_key, _tie64 = info
        return int(len(ids))

    total_freed_pages = 0
    for round_idx in range(max_rounds):
        if _time_exceeded():
            break

        before_inuse = int(page_allocator.get_num_inuse_pages())
        if before_inuse <= target_num_pages:
            break

        pages_needed = before_inuse - int(target_num_pages)
        allocated_by_page = _get_allocated_by_page()
        if not allocated_by_page:
            break

        postorder_nodes = _iter_nodes_postorder()
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
                total += int(subtree_tokens.get(child, 0))
                if subtree_has_lock.get(child, False):
                    has_lock = True

            subtree_tokens[node] = int(total)
            subtree_has_lock[node] = bool(has_lock)

        # Compute per-page coverage and locked pages.
        locked_pages: set[int] = set()
        known_blocks_by_page: dict[int, int] = defaultdict(int)
        nodes_by_page: dict[int, list[Any]] = defaultdict(list)

        for node in postorder_nodes:
            info = _get_node_info(node)
            if info is None:
                continue
            _ids, counts, _sort_key, _tie64 = info
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
                    (_get_node_info(n) or ([], {}, (), 0))[2],
                    (_get_node_info(n) or ([], {}, (), 0))[3],
                )
            )

            cost_tokens = 0
            for r in roots:
                cost_tokens += int(subtree_tokens.get(r, 0))

            candidate_pages.append((int(cost_tokens), int(allocated), int(pid), roots))

        if not candidate_pages:
            break

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
                (_get_node_info(n) or ([], {}, (), 0))[2],
                (_get_node_info(n) or ([], {}, (), 0))[3],
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

        # Mark nodes inside the selected subtrees for fast membership checks.
        in_target: dict[Any, bool] = {}
        root = getattr(cache, "root_node", None)
        stack2: list[tuple[Any, bool]] = [(root, False)] if root is not None else []
        visited2: set[Any] = set()
        while stack2:
            node, active = stack2.pop()
            if node is None or node in visited2:
                continue
            visited2.add(node)
            active2 = active or (node in minimal_set)
            in_target[node] = bool(active2)

            children = getattr(node, "children", None)
            if isinstance(children, dict):
                items = list(children.items())
                items.sort(key=lambda kv: _stable_child_key(kv[0]))
                for _k, child in reversed(items):
                    stack2.append((child, active2))
            elif children is not None:
                try:
                    child_list = list(children)
                except Exception:
                    child_list = []
                child_list.sort(key=lambda c: (type(c).__name__, repr(c)))
                for child in reversed(child_list):
                    stack2.append((child, active2))

        leaves = getattr(cache, "_collect_leaves", lambda: [])()
        pushed: set[Any] = set()
        candidates: list[tuple[tuple[int, ...], int, Any]] = []

        def _push(node: Any) -> None:
            if node in pushed:
                return
            if not in_target.get(node, False):
                return
            info = _get_node_info(node)
            if info is None:
                return
            _ids, _counts, sort_key, tie64 = info
            heapq.heappush(candidates, (sort_key, tie64, node))
            pushed.add(node)

        for node in leaves:
            _push(node)

        num_nodes_evicted = 0
        while remaining and candidates and not _time_exceeded():
            _sort_key, _tie64, node = heapq.heappop(candidates)

            if not in_target.get(node, False):
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
            ids, counts, _sort_key2, _tie64b = info

            cache.token_to_kv_pool_allocator.free(ids)
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
                and in_target.get(p, False)
                and int(getattr(p, "lock_ref", 1) or 0) == 0
                and len(getattr(p, "children", {})) == 0
            ):
                _push(p)
                p = getattr(p, "parent", None)

        after_inuse = int(page_allocator.get_num_inuse_pages())
        freed = before_inuse - after_inuse
        if rank == 0:
            logger.debug(
                f"[shrink-evict] round={round_idx} before_inuse={before_inuse} "
                f"after_inuse={after_inuse} target={target_num_pages} "
                f"selected_pages={sorted(selected_pages_set)} freed_pages={freed} "
                f"evicted_nodes={num_nodes_evicted}"
            )

        if freed <= 0 and num_nodes_evicted == 0:
            break
        total_freed_pages += int(freed)

    return int(total_freed_pages)


class ElasticAllocatorPatch(VersionAwarePatch, BasePatch):
    """Inject ElasticTokenToKVPoolAllocator into SGLang's allocator module"""

    library = "sglang"
    target_module = "sglang.srt.mem_cache.allocator"
    patch_name = "elastic_allocator"

    def apply(self, alloc_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        # Apply version-specific patches
        success = self.inject_elastic_allocator(alloc_mod)
        if success:
            success &= self.alias_allocator_to_elastic(alloc_mod)
        return success

    @version_range(SGLANG_ALL_RANGE)
    def inject_elastic_allocator(self, alloc_mod: types.ModuleType) -> bool:
        """Inject ElasticTokenToKVPoolAllocator"""
        if hasattr(alloc_mod, "ElasticTokenToKVPoolAllocator"):
            self.logger.debug("ElasticTokenToKVPoolAllocator already exists")
            return True

        try:
            import torch

            BaseTokenToKVPoolAllocator = getattr(alloc_mod, "BaseTokenToKVPoolAllocator")

            class ElasticTokenToKVPoolAllocator(
                BaseTokenToKVPoolAllocator  # type: ignore[misc, valid-type]
            ):
                def __init__(self, size: int, dtype, device: str, kvcache, *args, **kwargs) -> None:
                    super().__init__(size, 1, dtype, device, kvcache, *args, **kwargs)
                    if not hasattr(kvcache, "kvcached_allocator"):
                        raise ValueError("ElasticTokenToKVPoolAllocator requires elastic MHA pool")
                    if "cuda" not in device:
                        raise ValueError("ElasticTokenToKVPoolAllocator only supports cuda device")
                    self.kvcached_allocator = kvcache.kvcached_allocator

                def available_size(self):
                    return self.kvcached_allocator.available_size()

                def alloc(self, need_size: int):
                    indices = self.kvcached_allocator.alloc(need_size)
                    if indices is None:
                        # Match SGLang allocator contract: return None on OOM so callers
                        # can raise a proper error message (instead of crashing here).
                        return None
                    return torch.tensor(indices, dtype=torch.int32, device="cuda")

                def free(self, free_index):
                    if self.is_not_in_free_group:
                        try:
                            indices: list[int] = free_index.cpu().numpy().tolist()
                        except Exception:
                            indices = list(free_index)
                        return self.kvcached_allocator.free(indices)
                    else:
                        self.free_group.append(free_index)

                def clear(self):
                    if hasattr(self, "kvcached_allocator"):
                        self.kvcached_allocator.clear()

            setattr(alloc_mod, "ElasticTokenToKVPoolAllocator", ElasticTokenToKVPoolAllocator)
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject ElasticTokenToKVPoolAllocator: {e}")
            return False

    @version_range(SGLANG_ALL_RANGE)
    def alias_allocator_to_elastic(self, alloc_mod: types.ModuleType) -> bool:
        """Alias TokenToKVPoolAllocator to ElasticTokenToKVPoolAllocator"""
        if self._is_already_patched(alloc_mod, "__kvcached_allocator_aliased__"):
            return True

        try:
            ElasticTokenToKVPoolAllocator = getattr(alloc_mod, "ElasticTokenToKVPoolAllocator")
            if ElasticTokenToKVPoolAllocator is None:
                return False
            alloc_mod.TokenToKVPoolAllocator = ElasticTokenToKVPoolAllocator  # type: ignore
            self._mark_as_patched(alloc_mod, "__kvcached_allocator_aliased__")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to alias allocator to elastic one: {e}")
            return False


class ElasticMemoryPoolPatch(VersionAwarePatch, BasePatch):
    """Inject ElasticMHATokenToKVPool into SGLang's memory pool module"""

    library = "sglang"
    target_module = "sglang.srt.mem_cache.memory_pool"
    patch_name = "elastic_memory_pool"

    def apply(self, mem_pool_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        # Apply version-specific patches
        success = self.inject_elastic_mem_pool(mem_pool_mod)
        if success:
            success &= self.alias_mem_pool_to_elastic(mem_pool_mod)
        return success

    @version_range(SGLANG_ALL_RANGE)
    def inject_elastic_mem_pool(self, mem_pool_mod: types.ModuleType) -> bool:
        """Inject ElasticMHATokenToKVPool"""
        if hasattr(mem_pool_mod, "ElasticMHATokenToKVPool"):
            self.logger.debug("ElasticMHATokenToKVPool already exists")
            return True

        try:
            MHATokenToKVPool = getattr(mem_pool_mod, "MHATokenToKVPool")

            class ElasticMHATokenToKVPool(MHATokenToKVPool):  # type: ignore
                def __init__(
                    self,
                    size: int,
                    page_size: int,
                    dtype,
                    head_num: int,
                    head_dim: int,
                    layer_num: int,
                    device: str,
                    enable_memory_saver: bool,
                    start_layer: Union[int, None] = None,
                    end_layer: Union[int, None] = None,
                    *args,
                    **kwargs,
                ) -> None:
                    super().__init__(
                        size=size,
                        page_size=page_size,
                        dtype=dtype,
                        head_num=head_num,
                        head_dim=head_dim,
                        layer_num=layer_num,
                        device=device,
                        enable_memory_saver=enable_memory_saver,
                        start_layer=start_layer,
                        end_layer=end_layer,
                        *args,
                        **kwargs,
                    )
                    import kvcached.integration.sglang.interfaces as kvi

                    self.cell_size = self.head_num * self.head_dim * dtype.itemsize
                    self.kvcached_allocator = kvi.get_kv_cache_manager(
                        size + page_size, page_size, self.cell_size, layer_num
                    )

                    k_size, v_size = self.get_kv_size_bytes()
                    GB = 1024**3
                    k_size_phy, v_size_phy = self.get_kv_size_bytes_phy()

                    logger.info(
                        f"VirtualKV Cache is allocated. #tokens: {size}, K size: "
                        f"{k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
                    )
                    logger.info(
                        f"Physical KV Cache limits by --mem-fraction-static: "
                        f"#tokens: {size}, K size: "
                        f"{k_size_phy / GB:.2f} GB, V size: {v_size_phy / GB:.2f} GB"
                    )

                    self.mem_usage = (k_size + v_size) / GB

                def __del__(self):  # best-effort cleanup
                    try:
                        import kvcached.integration.sglang.interfaces as kvi

                        kvi.shutdown_kvcached()
                    except Exception:
                        pass

                def _create_buffers(self):
                    import kvcached.integration.sglang.interfaces as kvi

                    # Initialize kvcached with overlap scheduling to be conservative
                    kvi.init_kvcached(async_sched=True)

                    if "cuda" not in self.device:
                        raise ValueError("ElasticMHATokenToKVPool only supports cuda device")
                    self.k_buffer, self.v_buffer = kvi.alloc_kv_cache(
                        kvcache_shape=(
                            self.size + self.page_size,
                            self.head_num,
                            self.head_dim,
                        ),
                        dtype=self.dtype,
                        device=self.device,
                        num_layers=self.layer_num,
                        page_size=self.page_size,
                        attention_type="MHA",
                        kv_layout="NHD",
                    )

                def get_kv_size_bytes_phy(self):
                    """Return the physical memory limits of the K/V buffers.

                    This limit is enforced by `--mem-fraction-static` option.
                    """
                    total_tokens = self.size + self.page_size
                    elems_per_token = self.head_num * self.head_dim
                    bytes_per_elem = self.dtype.itemsize

                    k_size_bytes = self.layer_num * total_tokens * elems_per_token * bytes_per_elem
                    v_size_bytes = k_size_bytes

                    return k_size_bytes, v_size_bytes

            setattr(mem_pool_mod, "ElasticMHATokenToKVPool", ElasticMHATokenToKVPool)
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject ElasticMHATokenToKVPool: {e}")
            return False

    @version_range(SGLANG_ALL_RANGE)
    def alias_mem_pool_to_elastic(self, mem_pool_mod: types.ModuleType) -> bool:
        """Alias MHATokenToKVPool to ElasticMHATokenToKVPool"""
        if self._is_already_patched(mem_pool_mod, "__kvcached_mempool_aliased__"):
            return True

        try:
            ElasticMHATokenToKVPool = getattr(mem_pool_mod, "ElasticMHATokenToKVPool")
            if ElasticMHATokenToKVPool is None:
                return False
            # Alias defaults so core code will use elastic variants
            mem_pool_mod.MHATokenToKVPool = ElasticMHATokenToKVPool  # type: ignore
            self._mark_as_patched(mem_pool_mod, "__kvcached_mempool_aliased__")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to alias memory_pool to elastic one: {e}")
            return False


class SchedulerMemoryLeakPatch(VersionAwarePatch, BasePatch):
    """Patch SGLang scheduler to suppress memory leak check when kvcached is enabled"""

    library = "sglang"
    target_module = "sglang.srt.managers.scheduler"
    target_class = "Scheduler"
    patch_name = "scheduler_memory_leak"

    def apply(self, sched_mod: types.ModuleType) -> bool:
        # Initialize version info
        if not self.initialize_version_info():
            return False

        # Apply version-specific patches
        return self.patch_scheduler_memory_leak(sched_mod)

    @version_range(SGLANG_ALL_RANGE)
    def patch_scheduler_memory_leak(self, sched_mod: types.ModuleType) -> bool:
        """Patch scheduler to suppress memory leak check when kvcached is enabled"""
        Scheduler = self._get_target_class(sched_mod)
        if Scheduler is None:
            return False

        target_method_name: Union[str, None] = None
        for name, fn in inspect.getmembers(Scheduler, predicate=inspect.isfunction):
            try:
                src = inspect.getsource(fn)
            except Exception:
                continue
            if "token_to_kv_pool_allocator memory leak detected!" in src or (
                "memory leak detected" in src and "token_to_kv_pool_allocator" in src
            ):
                target_method_name = name
                break

        if target_method_name is None:
            self.logger.debug("No memory leak detection method found in Scheduler")
            return False

        original = getattr(Scheduler, target_method_name)
        if self._is_already_patched(original):
            self.logger.debug("Scheduler memory leak check already patched")
            return True

        def _wrapped(self, *args: Any, **kwargs: Any):
            # Disable memory leak detection when ENABLE_KVCACHED is set
            if enable_kvcached():
                return
            return original(self, *args, **kwargs)

        self._mark_as_patched(_wrapped)
        setattr(Scheduler, target_method_name, _wrapped)
        return True


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

        self._mark_as_patched(RadixCache, "__kvcached_shrink_eviction__")
        return True


class ResizeBeforeEvictPatch(VersionAwarePatch, BasePatch):
    """Apply external resize target before SGLang eviction runs.

    SGLang calls `evict_from_tree_cache()` before `allocator.alloc()`. In the
    kvcached integration, applying a shrink inside `KVCacheManager.alloc()` can
    happen after SGLang's eviction, which may re-introduce OOM.

    This patch forces kvcached to apply the pending external resize target (if
    any) before SGLang decides whether/how much to evict.
    """

    library = "sglang"
    target_module = "sglang.srt.mem_cache.common"
    patch_name = "resize_before_evict"

    def apply(self, common_mod: types.ModuleType) -> bool:
        if not self.initialize_version_info():
            return False
        return self.patch_common(common_mod)

    @version_range(SGLANG_ALL_RANGE)
    def patch_common(self, common_mod: types.ModuleType) -> bool:
        original = getattr(common_mod, "evict_from_tree_cache", None)
        if original is None:
            return False
        if self._is_already_patched(original):
            return True

        def _wrapped(tree_cache, num_tokens: int):
            if enable_kvcached() and tree_cache is not None:
                allocator = getattr(tree_cache, "token_to_kv_pool_allocator", None)
                kvcached_allocator = getattr(allocator, "kvcached_allocator", None)
                apply_fn = getattr(kvcached_allocator, "maybe_apply_resize_target", None)
                if callable(apply_fn):
                    try:
                        apply_fn()
                    except Exception:
                        # Best-effort; fall back to the original behavior.
                        pass
            return original(tree_cache, num_tokens)

        self._mark_as_patched(_wrapped)
        setattr(common_mod, "evict_from_tree_cache", _wrapped)
        return True
