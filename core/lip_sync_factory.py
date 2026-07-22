from __future__ import annotations

import asyncio

from config.config import config as service_config


def create_lip_sync_manager(vae_idle_event: asyncio.Event):
    """Construct the configured video backend without importing unused GPU stacks."""

    if service_config.video.backend == "ti2v5b_musetalk":
        from core.wanmuse.manager import WanMuseLipSyncManager

        return WanMuseLipSyncManager(vae_idle_event=vae_idle_event)

    from core.lip_sync import LipSyncManager

    return LipSyncManager(vae_idle_event=vae_idle_event)
