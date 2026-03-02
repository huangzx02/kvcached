# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os
import types

from wrapt.importer import when_imported

from kvcached.integration.patch_base import PatchManager, log_patch_results
from kvcached.integration.sglang.patches import (
    SGLANG_ALL_RANGE,
    ElasticAllocatorPatch,
    ElasticMemoryPoolPatch,
    ElasticMLAMemoryPoolPatch,
    SchedulerMemoryLeakPatch,
)
from kvcached.integration.sglang.hiradix_cache_patches import HiRadixCacheShrinkEvictionPatch
from kvcached.integration.sglang.kv_hotness_patches import ScheduleBatchKVHotnessPatch
from kvcached.integration.sglang.radix_cache_patches import RadixCacheShrinkEvictionPatch
from kvcached.integration.sglang.scheduler_patches import SchedulerShrinkRetractPatch
from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


def _env_enabled() -> bool:
    return os.getenv("KVCACHED_AUTOPATCH", "false").lower() in ("true", "1")


@when_imported("sglang")
def _patch_sglang(_sglang: types.ModuleType) -> None:
    if not _env_enabled():
        logger.debug("Disabled by KVCACHED_AUTOPATCH")
        return
    logger.debug("Patching SGLang")

    # Create patch manager and register version-specific SGLang patches
    patch_manager = PatchManager("sglang")

    patch_manager.register_patches_with_versions(
        [
            (ElasticAllocatorPatch(), SGLANG_ALL_RANGE),
            (ElasticMemoryPoolPatch(), SGLANG_ALL_RANGE),
            (ElasticMLAMemoryPoolPatch(), SGLANG_ALL_RANGE),
            (SchedulerMemoryLeakPatch(), SGLANG_ALL_RANGE),
            (RadixCacheShrinkEvictionPatch(), SGLANG_ALL_RANGE),
            (HiRadixCacheShrinkEvictionPatch(), SGLANG_ALL_RANGE),
            (SchedulerShrinkRetractPatch(), SGLANG_ALL_RANGE),
            (ScheduleBatchKVHotnessPatch(), SGLANG_ALL_RANGE),
        ]
    )

    # Apply all patches
    results = patch_manager.apply_all_patches()

    # Log results
    log_patch_results("sglang", results)
