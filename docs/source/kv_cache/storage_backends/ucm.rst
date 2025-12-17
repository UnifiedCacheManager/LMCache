UCM Backend
===========

.. _nixl-overview:

Overview
--------

The UCM (Unified Cache Manager) backend integrates UCM’s existing store implementations—such as LocalStore, 
NFSStore, 3FSStore—into LMCache to enable efficient prefix caching. It leverages UCM’s unified block ID + offset 
addressing mechanism to manage KV cache data, allowing LMCache to transparently persist and retrieve cached KV blocks 
from arbitrary external storage systems—including local filesystems, NFS, S3, and more—without 
modifying upper-layer logic. By plugging UCM stores directly into LMCache’s storage interface, this backend enables 
cross-request, cross-session, and even cross-node reuse of KV caches, significantly reducing redundant computation and 
GPU memory consumption in long-context LLM inference scenarios.

Prerequisites
~~~~~~~~~~~~~
- **LMCache**: Install with ``pip install lmcache``
- **UCM**: Install from `UCM GitHub repository <https://github.com/ModelEngine-Group/unified-cache-management>`_

Example Configurations
----------------------

**Step 1: Create Configuration File**

Create your ``ucm-config.yaml``:

.. code-block:: yaml
    
    chunk_size: 256
    local_cpu: False
    max_local_cpu_size: 10
    remote_url: "ucmstore://localhost:0/home/ucm-test"
    save_unfull_chunk: False
    extra_config:
      store_name: "UcmNfsStore"
      save_chunk_meta: False
      use_odirect: False
      stream_number: 32
      buffer_number: 0

**Step 2: Start vLLM service**

.. code-block:: bash

    PYTHONHASHSEED=123456 \
    LMCACHE_CONFIG_FILE="ucm-config.yaml" \
    vllm serve QwQ-32B \
        --max-model-len 20000 \
        --host 0.0.0.0 \
        --port 8000 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

**Step 3: Verify the Setup**

.. code-block:: bash

    curl -X POST "http://localhost:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
          "model": "QwQ-32B",
          "prompt": "Hello, world!",
          "max_new_tokens": 100,
          "temperature": 0.7
        }'


Configuration Parameters
------------------------

LMCache Parameters
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``chunk_size``
     - 256
     - Number of tokens per KV chunk
   * - ``remote_url``
     - Required
     - UcmStore data storage path  (format: ``ucmstore://localhost:0/home/ucm-test``)
   * - ``local_cpu``
     - False
     - Enable/disable local CPU caching (set to False for pure UCM evaluation)
   * - ``max_local_cpu_size``
     - Required
     - Maximum local CPU cache size in GB (required even when local_cpu is False)
   * - ``save_unfull_chunk``
     - Required
     - Save unfull chunks (must be False for UCM)

UCM Parameters (in extra_config)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``store_name``
     - "UcmNfsStore"
     - UCM store name
   * - ``save_chunk_meta``
     - Required
     - Whether to save chunk metadata in the UCM store along with your data(must be False for UCM)
   * - ``use_odirect``
     - False
     - Whether to use O_DIRECT for file I/O
   * - ``stream_number``
     - 32
     - Number of streams for UCM
   * - ``buffer_number``
     - Required
     - Number of buffers for UCM (must be 0 to rely on LMCache local_cpu as the buffer)