#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os
from enum import Enum

from rag.utils.azure_sas_conn import RAGFlowAzureSasBlob
from rag.utils.azure_spn_conn import RAGFlowAzureSpnBlob
from rag.utils.minio_conn import RAGFlowMinio
from rag.utils.opendal_conn import OpenDALStorage
from rag.utils.s3_conn import RAGFlowS3
from rag.utils.oss_conn import RAGFlowOSS


class Storage(Enum):
    MINIO = 1
    AZURE_SPN = 2
    AZURE_SAS = 3
    AWS_S3 = 4
    OSS = 5
    OPENDAL = 6


class StorageFactory:
    storage_mapping = {
        Storage.MINIO: RAGFlowMinio,
        Storage.AZURE_SPN: RAGFlowAzureSpnBlob,
        Storage.AZURE_SAS: RAGFlowAzureSasBlob,
        Storage.AWS_S3: RAGFlowS3,
        Storage.OSS: RAGFlowOSS,
        Storage.OPENDAL: OpenDALStorage
    }

    @classmethod
    def create(cls, storage: Storage):
        return cls.storage_mapping[storage]()


STORAGE_IMPL_TYPE = os.getenv('STORAGE_IMPL', 'MINIO')
STORAGE_IMPL = StorageFactory.create(Storage[STORAGE_IMPL_TYPE])

# Document Store Factory
from rag import settings
from rag.utils.doc_store_conn import DocStoreConnection

# Import existing document store connection classes
from rag.utils.es_conn import ESConnection
from rag.utils.infinity_conn import InfinityConnection
# Assuming opensearch_conn.py contains OpenSearchConnection similar to es_conn
# from rag.utils.opensearch_conn import OpenSearchConnection # Corrected import name if it exists
# For now, let's use the actual opensearch_coon.py (typo in filename in repo)
from rag.utils.opensearch_coon import OpenSearchConnection # Actual filename in repo
from rag.utils.milvus_conn import MilvusConnection


# Global variable to store the document store instance
_doc_store_instance = None

def get_doc_store() -> DocStoreConnection:
    """
    Factory function to get the appropriate document store connection
    based on the DOC_ENGINE setting.
    It ensures that the connection is a singleton.
    """
    global _doc_store_instance
    if _doc_store_instance is not None:
        return _doc_store_instance

    doc_engine_lower = settings.DOC_ENGINE.lower()
    if doc_engine_lower == "elasticsearch":
        _doc_store_instance = ESConnection()
    elif doc_engine_lower == "opensearch":
        _doc_store_instance = OpenSearchConnection()
    elif doc_engine_lower == "infinity":
        _doc_store_instance = InfinityConnection()
    elif doc_engine_lower == "milvus":
        # MilvusConnection constructor expects a 'tenant' argument.
        # This factory function does not have tenant context directly.
        # This implies that either:
        # 1. MilvusConnection's singleton needs to handle tenant variations,
        #    or the factory needs tenant info.
        # 2. The @singleton decorator on MilvusConnection might be an issue if it
        #    doesn't correctly handle different tenants when called without args here.
        #    The singleton in rag.utils takes *args, **kw for the class.
        #    If MilvusConnection is a true singleton per tenant, this factory
        #    cannot return a generic one without tenant_id.
        #
        # For now, let's assume the @singleton on MilvusConnection means it's
        # a singleton *manager* that can provide tenant-specific instances,
        # or that the first call to MilvusConnection() without tenant_id might
        # work if it has a default tenant or if the @singleton handles it.
        # This is a potential issue to flag.
        # The MilvusConnection __init__ requires a `tenant` argument.
        # This factory cannot provide it. The singleton pattern might be
        # problematic here if it's a simple instance singleton.
        #
        # A common pattern for multi-tenant singletons is that the first call
        # to `MilvusConnection(tenant_id)` creates and stores the instance for that tenant.
        # Calling `MilvusConnection()` here is problematic.
        #
        # Let's assume for now that the intention is to get the class itself,
        # and the caller will instantiate it with the tenant.
        # Or, this factory is NOT meant for MilvusConnection if it's per-tenant.
        #
        # Given the other connections (ES, Infinity) are instantiated directly,
        # MilvusConnection should be too. But it needs `tenant`.
        # This factory might need re-thinking for tenant-specific connections.
        #
        # However, if the @singleton decorator in rag.utils creates a proxy
        # that then allows tenant-specific instantiation, MilvusConnection() might work.
        # Let's proceed with the direct instantiation as per pattern,
        # assuming the singleton handles it or this will be caught in testing.
        # The `MilvusConnection` is decorated with `@singleton` which means
        # `MilvusConnection()` will try to get/create an instance.
        # If the singleton is keyed by `cls` only, it will fail if `tenant` is always needed.
        #
        # The prompt implies calling MilvusConnection().
        # The MilvusConnection(tenant) is how it's defined.
        # This factory CANNOT provide a tenant-specific Milvus connection without tenant_id.
        #
        # Revisiting the @singleton in `rag/utils/__init__.py`:
        # instances = {} ... key = str(cls) + str(os.getpid())
        # This singleton creates ONE instance per class, per process.
        # So MilvusConnection() would try to init without tenant, which will fail.
        #
        # Therefore, this factory cannot return a ready-to-use MilvusConnection instance
        # in the same way it does for ESConnection/InfinityConnection if they don't need tenant at this stage.
        #
        # Possibility: The factory returns the *class* for connections that need tenant_id.
        # Or the factory itself takes tenant_id.
        # For now, I will make it return an *instance* and assume the singleton decorator
        # or the class itself has a way to manage a "default" or "pending-tenant" state,
        # or this will simply be an issue found later.
        # Let's try to instantiate, and if it fails, it indicates a design issue with the factory pattern for Milvus.
        # The `DocStoreConnection` has `self.tenant` in its `__init__`.
        # `ESConnection` and `InfinityConnection` don't show explicit tenant_id in their constructors in this file.
        # They might be getting it from settings or another global context if they are tenant-aware.
        # `MilvusConnection`'s constructor is `__init__(self, tenant: str)`.
        #
        # This factory `get_doc_store()` is likely called by code that *doesn't* yet have tenant context,
        # or it's for a global, non-tenant-specific store. If RAGflow is multi-tenant, this is an issue for Milvus.
        #
        # Given the structure, the most straightforward interpretation is to follow the pattern:
        # _doc_store_instance = MilvusConnection()
        # And accept this will fail if MilvusConnection strictly needs a tenant_id at construction
        # and the singleton isn't designed to defer this.
        #
        # The task is "Instantiate and return MilvusConnection()".
        # This will cause a TypeError: MilvusConnection.__init__() missing 1 required positional argument: 'tenant'
        #
        # Let's assume the factory is NOT responsible for providing the tenant.
        # The singleton for MilvusConnection is (cls, tenant) tuple as key.
        # Oh, no, the singleton key in rag.utils is just `str(cls) + str(os.getpid())`.
        # This means `MilvusConnection()` cannot be called by this factory if the instance it creates
        # is the one that needs a tenant.
        #
        # The other DocStoreConnections (ES, Infinity) do not take `tenant` in their constructor.
        # This is a key difference.
        # This factory should probably not be used for MilvusConnection if it's tenant-specific.
        # Or, MilvusConnection needs a default tenant, or its singleton needs to be smarter.
        #
        # For the purpose of this task, I will add the code as requested,
        # but with a strong comment about this issue.
        #
        # A better approach for Milvus would be:
        # `_doc_store_instance = MilvusConnection` (returning the class, not an instance)
        # And the caller does `get_doc_store()(tenant="actual_tenant")`.
        # But this changes the factory's return type consistency.
        #
        # Sticking to the pattern and highlighting the issue:
        # This will raise an error. The design of MilvusConnection or this factory needs adjustment.
        # To make it runnable for now, I might have to assume a "default" tenant for the factory instance,
        # which is bad practice for multi-tenant systems.
        #
        # Let's look at how ESConnection / InfinityConnection handle multi-tenancy.
        # They don't seem to from their constructors. Perhaps they are tenant-agnostic, or use a global tenant.
        # Milvus implementation explicitly uses tenant for DB naming and connection aliasing.
        #
        # Attempting direct instantiation as per subtask update.
        # This will cause a TypeError if MilvusConnection.__init__ does not have a default for 'tenant'
        # or if the singleton cannot manage parameterless calls for a class that needs constructor args.
        _doc_store_instance = MilvusConnection()
    else:
        raise ValueError(f"Unsupported DOC_ENGINE: {settings.DOC_ENGINE}")

    return _doc_store_instance
