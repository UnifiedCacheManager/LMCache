# SPDX-License-Identifier: Apache-2.0
# Standard
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, no_type_check
import asyncio
import os

# Third Party
import aiofiles
import aiofiles.os

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from vllm.distributed.parallel_state import get_world_group
from ucm.store.factory import UcmConnectorFactory

logger = init_logger(__name__)

METADATA_BYTES_LEN = 28

class UcmstoreConnector(RemoteConnector):
    """Connector for ucmstore remote storage."""

    def __init__(
        self,
        base_paths_str: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: Optional[LMCacheEngineConfig],
    ):
        """
        Args:
            base_paths_str: Comma separated storage paths
            loop: Asyncio event loop
            local_cpu_backend: Memory allocator interface
            config: Lmcache engine config
        """
        # Parse comma separated paths
        self.base_paths = (
            [Path(p.strip()) for p in base_paths_str.split(",")]
            if "," in base_paths_str
            else [Path(base_paths_str)]
        )

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.config = config

    def init_chunk_meta(
            self,
            config: Optional[LMCacheEngineConfig],
            metadata: Optional[LMCacheEngineMetadata],
        ) -> None:
        super().init_chunk_meta(config, metadata)
        self.role = metadata.role
        self._init_store()

    def _init_store(self):
        config = self.config.extra_config
        config["storage_backends"] = ":".join(str(p) for p in self.base_paths)
        if self.save_chunk_meta:
            config["kv_block_size"] = self.full_chunk_size + METADATA_BYTES_LEN
        else:
            config["kv_block_size"] = self.full_chunk_size
        config["role"] = self.role
        config["device"] = get_world_group().local_rank
        config["io_size"] = config["kv_block_size"]
        config['use_direct'] = config.get('use_odirect', False)

        self.store = UcmConnectorFactory.create_connector(
            config.get("store_name", "UcmNfsStore"),
            config,
        )
        
    def _block_id(self, key: CacheEngineKey) -> str:
        return hashlib.blake2b(
            key.to_string().encode("utf-8"), digest_size=16
        ).hexdigest()
    
    async def exists(self, key: CacheEngineKey) -> bool:
        block_id = self._block_id(key) 
        rets = await asyncio.to_thread(self.store.lookup, [block_id])
        return bool(rets[0])

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Check if key exists in ucmstore synchronized"""
        block_id = self._block_id(key)
        is_exists = self.store.lookup([block_id])[0]
        return is_exists
    
    async def _get_without_metadata(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """load kv data without metadata form ucmstore"""
        memory_obj = self.local_cpu_backend.allocate(self.meta_shape, self.meta_dtype, self.meta_fmt)
        if memory_obj is None:
            logger.warning("allocate host buffer failed")
            return None

        block_id = self._block_id(key)
        tensor = memory_obj.tensor
        task = self.store.fetch_data(
            [block_id],
            [0],
            [tensor.data_ptr()],
            [tensor.numel() * tensor.element_size()],
        )

        ret = await asyncio.to_thread(self.store.wait, task)
        if ret != 0:
            logger.error(f"ucmstore load block {block_id} failed for key {key.to_string()}, ret={ret}")
            memory_obj.ref_count_down()
            return None

        return memory_obj

    async def _get_with_metadata(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """TODO: load kv data and metadata form ucmstore"""
        pass

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """load kv data from ucmstore"""
        if self.save_chunk_meta:
            mem_obj = await self._get_with_metadata(key)
        else:
            mem_obj = await self._get_without_metadata(key)
        return mem_obj
    
    async def _put_without_metadata(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        block_id = self._block_id(key)
        tensor = memory_obj.tensor

        ret = self.store.create([block_id])[0]
        if ret != 0:
            logger.error(f"ucmstore create block {block_id} for key {key.to_string()} failed, ret code:{ret}")
            return

        task = self.store.dump_data(
            [block_id], [0],
            [tensor.data_ptr()],
            [tensor.numel() * tensor.element_size()],
        )

        ret = await asyncio.to_thread(self.store.wait, task)
        if ret != 0:
            logger.error(f"ucmstore dump block {block_id} failed for key {key.to_string()}, ret={ret}")

        self.store.commit([block_id], ret == 0)

    async def _put_with_metadata(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """TODO: dump kv data and metadata to ucmstore"""
        pass

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """save kv data to ucmstore"""
        if self.save_chunk_meta:
            await self._put_with_metadata(key, memory_obj)
        else:
            await self._put_without_metadata(key, memory_obj)
    
    def remove_sync(self, key: CacheEngineKey) -> bool:
        """ucmstore does not support delete operation"""
        return False

    @no_type_check
    async def list(self) -> List[str]:
        """ucmstore does not support listing all keys"""
        return []
    
    async def close(self):
        """Clean up resources"""
        logger.info("Closed the ucmstore connector")
