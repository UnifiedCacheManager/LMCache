# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    parse_remote_url,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class UcmStoreConnectorAdapter(ConnectorAdapter):
    """Adapter for Filesystem connectors."""

    def __init__(self) -> None:
        super().__init__("ucmstore://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .ucm_connector import UcmstoreConnector

        logger.info(f"Creating ucmstore connector for URL: {context.url}")
        parse_url = parse_remote_url(context.url)

        return UcmstoreConnector(
            parse_url.path, context.loop, context.local_cpu_backend, context.config
        )
