import copy
# import json # Not used
import logging
# import os # Not used
# import re # Not used
# import time # Not used

import pandas as pd # Keep for sql's potential future type hint
import pymilvus
from pymilvus import DataType # Explicitly import DataType
from rag.settings import settings
from rag.utils.doc_store_conn import DocStoreConnection # Comparison, Logical not directly used by Milvus specifics
from rag.utils import singleton


@singleton
class MilvusConnection(DocStoreConnection):
    def __init__(self, tenant: str):
        super().__init__(tenant)
        self.conn = self._create_connection()
        # Sanitize tenant ID for db name and apply length considerations
        db_tenant_suffix = tenant.replace("-", "_")
        self.db_name = f"rag_{db_tenant_suffix}"
        # Milvus DB names max length can be an issue, e.g. 63 chars.
        if len(self.db_name) > 63:
            logging.warning(f"Database name {self.db_name} based on tenant {tenant} is longer than 63 chars. This might lead to errors. Consider shortening/hashing tenant IDs for DB names.")
            # Potentially truncate or hash: e.g., self.db_name = self.db_name[:63]
        self._create_db_if_not_exists()

    def _create_connection(self):
        # TODO(developer): make this configurable in settings.py, needs MILVUS config in settings
        # For now, using default Milvus connection parameters
        try:
            milvus_config = settings.MILVUS
            host = milvus_config.get("host", "localhost")
            port = milvus_config.get("port", "19530")
            # TODO(developer): Add user/password from milvus_config if Milvus is secured
            # user = milvus_config.get("user", "")
            # password = milvus_config.get("password", "")
        except AttributeError: # Fallback if MILVUS config is not yet in settings
            logging.warning("MILVUS configuration not found in settings. Using default Milvus connection.")
            host = "localhost"
            port = "19530"
            # user = ""
            # password = ""

        logging.info(f"Attempting to connect to Milvus at {host}:{port} for tenant {self.tenant}")
        try:
            conn_params = {
                "alias": f"default_{self.tenant}",  # Use a unique alias per tenant
                "host": host,
                "port": port,
            }
            # if user and password:
            #     conn_params["user"] = user
            #     conn_params["password"] = password

            conn = pymilvus.connections.connect(**conn_params)
            logging.info(f"Successfully connected/reused connection to Milvus for tenant {self.tenant} with alias default_{self.tenant}")
            return conn
        except Exception as e:
            logging.error(f"Failed to connect to Milvus for tenant {self.tenant}: {e}")
            raise

    def _create_db_if_not_exists(self):
        try:
            # Use has_database if available (PyMilvus 2.4.2+), otherwise fallback or assume list_database behavior
            if not pymilvus.utility.has_database(self.db_name, using=f"default_{self.tenant}"):
            # Older versions: if self.db_name not in pymilvus.utility.list_database(using=f"default_{self.tenant}"):
                logging.info(f"Database {self.db_name} does not exist. Creating now for tenant {self.tenant}.")
                pymilvus.utility.create_database(self.db_name, using=f"default_{self.tenant}")
                logging.info(f"Database {self.db_name} created successfully for tenant {self.tenant}.")

            # db_name is passed to each Collection/utility call, so no explicit select_db for the alias needed here.
            logging.info(f"Using database {self.db_name} for tenant {self.tenant} (alias default_{self.tenant})")
        except Exception as e:
            logging.error(f"Failed to create or check database {self.db_name} for tenant {self.tenant}: {e}")
            raise

    @property
    def dbType(self) -> str:
        return "milvus"

    def health(self) -> bool:
        try:
            # A simple check could be to list databases or collections
            # or get server version/status if available directly.
            # For now, checking if the connection alias exists is a basic health check.
            pymilvus.connections.get_connection(alias=f"default_{self.tenant}")
            # More robust: try a lightweight operation e.g. list_collections
            pymilvus.utility.list_collections(using=f"default_{self.tenant}", db_name=self.db_name)
            return True
        except Exception as e:
            logging.error(f"Milvus health check failed for tenant {self.tenant}: {e}")
            return False

    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):
        from pymilvus import CollectionSchema, FieldSchema, Collection # DataType imported at top

        collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_") # Milvus names have restrictions
        vector_field_name = f"q_{vectorSize}_vec"

        # Use consistent indexExist signature
        if self.indexExist(indexName, knowledgebaseId):
            logging.info(f"Collection {collection_name} already exists for tenant {self.tenant}. Skipping creation.")
            return

        try:
            # Define fields based on infinity_mapping.json and common RAG needs
            # Max length for VARCHAR, adjust as needed. Milvus max is 65535.
            MAX_VARCHAR_LENGTH = 1024  # For general text fields
            ID_VARCHAR_LENGTH = 255   # For IDs

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=ID_VARCHAR_LENGTH, description="Primary key chunk ID"),
                FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=ID_VARCHAR_LENGTH, description="Knowledge base ID"),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=ID_VARCHAR_LENGTH, description="Document ID"),
                FieldSchema(name="docnm_kwd", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH, description="Document name keyword"),
                FieldSchema(name="content_ltks", dtype=DataType.VARCHAR, max_length=65535, description="Content long tokens"), # Use larger limit for content
                FieldSchema(name="name_tks", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH, description="Name tokens"), # Assuming 'name_tks' from prompt, was 'name_kwd' in mapping
                FieldSchema(name="important_kwd", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH, description="Important keywords"),
                FieldSchema(name="question_tks", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH, description="Question tokens"),
                FieldSchema(name="page_num_int", dtype=DataType.INT32, description="Page number"), # Assuming it's an integer
                FieldSchema(name="create_timestamp_flt", dtype=DataType.FLOAT, description="Creation timestamp"),
                FieldSchema(name=vector_field_name, dtype=DataType.FLOAT_VECTOR, dim=vectorSize, description="Query vector embedding")
            ]

            # TODO(developer): Add more fields from infinity_mapping.json as needed, ensuring type compatibility.
            # e.g., title_tks, content_sm_ltks, available_int etc.

            schema = CollectionSchema(fields=fields, description=f"Collection for {indexName} of KB {knowledgebaseId}", enable_dynamic_field=True) # Enable dynamic fields for flexibility

            collection = Collection(
                name=collection_name,
                schema=schema,
                using=f"default_{self.tenant}",
                db_name=self.db_name
            )
            logging.info(f"Collection {collection_name} created successfully for tenant {self.tenant}.")

            # Create index for the vector field
            # TODO(developer): Make index type and params configurable
            index_params = {
                "metric_type": "L2", # Or IP for cosine similarity
                "index_type": "IVF_FLAT", # Or HNSW, etc.
                "params": {"nlist": 128},
            }
            # For HNSW:
            # index_params = {
            #     "metric_type": "L2",
            #     "index_type": "HNSW",
            #     "params": {"M": 16, "efConstruction": 256},
            # }
            collection.create_index(field_name=vector_field_name, index_params=index_params)
            logging.info(f"Index created on field {vector_field_name} for collection {collection_name}.")

            collection.load()
            logging.info(f"Collection {collection_name} loaded.")

        except Exception as e:
            logging.error(f"Failed to create index {collection_name} for tenant {self.tenant}: {e}")
            raise

    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        """ Deletes a collection specific to a knowledge base. """
        collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")
        try:
            # Use consistent signature for indexExist
            if self.indexExist(indexName, knowledgebaseId):
                pymilvus.utility.drop_collection(
                    collection_name=collection_name,
                    using=f"default_{self.tenant}",
                    db_name=self.db_name
                )
                logging.info(f"Collection {collection_name} deleted successfully for tenant {self.tenant}.")
            else:
                logging.warning(f"Collection {collection_name} does not exist, cannot delete for tenant {self.tenant}.")
        except Exception as e:
            logging.error(f"Failed to delete collection {collection_name} for tenant {self.tenant}: {e}")
            raise

    def indexExist(self, indexName: str, knowledgebaseId: str = None) -> bool:
        """ Checks if a collection specific to a knowledge base exists.
            If knowledgebaseId is None, indexName is treated as the full collection name.
        """
        if knowledgebaseId:
            collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")
        else:
            collection_name = indexName # Assume indexName is the full collection name

        try:
            return pymilvus.utility.has_collection(
                collection_name=collection_name,
                using=f"default_{self.tenant}",
                db_name=self.db_name
            )
        except Exception as e:
            logging.error(f"Failed to check if index {collection_name} exists for tenant {self.tenant}: {e}")
            return False

    def search(
        self,
        selectFields: list[str],
        highlightFields: list[str], # Not implemented in this pass
        condition: dict,
        matchExprs: list, # List of MatchDenseExpr or MatchTextExpr (from rag.utils.doc_store_conn)
        orderBy: object, # OrderByExpr (not fully implemented in this pass)
        offset: int,
        limit: int,
        indexNames: str | list[str], # Typically one indexName for Milvus context
        knowledgebaseIds: list[str], # Can be multiple KBs within the same index type
        aggFields: list[str] = [], # Not implemented in this pass
        rank_feature: dict | None = None # Not implemented in this pass
    ) -> list[dict]: # Return list of dicts
        from pymilvus import Collection, DataType
        # For MatchExpr types
        from rag.utils.doc_store_conn import MatchDenseExpr, MatchTextExpr


        if not knowledgebaseIds:
            logging.warning("Search called with no knowledgebaseIds.")
            return []

        # For simplicity, this initial implementation will handle one indexName.
        # If indexNames is a list, we'll use the first one.
        # Production systems might need to loop or combine results from multiple index types.
        current_index_name = indexNames[0] if isinstance(indexNames, list) else indexNames

        results_list = []

        # TODO(developer): How to combine results if searching across multiple KBs (collections)?
        # Current approach: Search each KB's collection and concatenate.
        # This might need adjustments based on desired ranking (e.g., if limit is global).
        # For now, limit is applied per collection search.

        for kb_id in knowledgebaseIds:
            collection_name = f"{current_index_name}_{kb_id}".replace("-", "_")
            # Use consistent indexExist signature
            if not self.indexExist(current_index_name, kb_id):
                logging.warning(f"Collection {collection_name} does not exist for tenant {self.tenant}. Skipping search for this KB.")
                continue

            try:
                collection = Collection(name=collection_name, using=f"default_{self.tenant}", db_name=self.db_name)
                collection.load()

                # Construct filter expression from `condition`
                filter_expr_parts = []
                if condition:
                    for field, value in condition.items():
                        if isinstance(value, list):
                            # Assuming 'in' operator for lists, e.g., "kb_id in ['kb1', 'kb2']"
                            # Ensure values are quoted if strings
                            formatted_values = [f"'{v}'" if isinstance(v, str) else str(v) for v in value]
                            filter_expr_parts.append(f"{field} in [{', '.join(formatted_values)}]")
                        elif isinstance(value, str):
                            filter_expr_parts.append(f"{field} == '{value}'")
                        else: # int, float, bool
                            filter_expr_parts.append(f"{field} == {value}")

                # Ensure the current kb_id is part of the filter, if not already covered by `condition`
                # This is important if condition is generic and we are looping through specific kb_id collections.
                # However, collection_name already implies the kb_id. So, an additional kb_id filter
                # inside the expression for a collection that IS that kb_id is redundant.
                # The main use of `condition` would be for other metadata.
                # Example: if condition was {'user_tag': 'test'}, expr becomes "user_tag == 'test'"

                final_filter_expr = " and ".join(filter_expr_parts) if filter_expr_parts else ""
                logging.debug(f"Milvus search filter expression for {collection_name}: '{final_filter_expr}'")

                # Process matchExprs
                for expr in matchExprs:
                    if isinstance(expr, MatchDenseExpr):
                        # Vector search
                        vector_to_search = expr.data
                        # Determine vector field name from schema, or assume from expr if provided
                        # For now, find the first float vector field in the schema.
                        vec_field_name = None
                        for field_schema in collection.schema.fields:
                            if field_schema.dtype == DataType.FLOAT_VECTOR:
                                vec_field_name = field_schema.name
                                # TODO: check dimension if expr.data has explicit dimension
                                break
                        if not vec_field_name:
                            logging.error(f"No float vector field found in schema for {collection_name}. Skipping dense search.")
                            continue

                        # TODO(developer): make search_params (like nprobe, ef) configurable
                        # These depend on the index type used (e.g., IVF_FLAT, HNSW)
                        # Default for IVF_FLAT: {"nprobe": 10}
                        # Default for HNSW: {"ef": 64}
                        # Assuming IVF_FLAT with nlist=128 as per createIdx
                        search_params = {
                            "metric_type": "L2", # Or "IP" - should match index metric_type
                            "params": {"nprobe": 10}, # Example for IVF_FLAT
                        }

                        # Milvus search API uses `anns_field`, `param`, `limit`, `expr`, `output_fields`, `offset`
                        # Consistency_level can also be set.

                        # Handle offset for pagination: Milvus search `offset` parameter
                        # The `limit` in search is top_k. If offset is used, Milvus returns `limit` results *after* the offset.
                        # So, if user wants limit=10, offset=10 (page 2), Milvus needs limit=10, offset=10.

                        # Ensure selectFields are valid schema fields, add 'id' if not present for merging.
                        # Milvus returns distance in results.
                        # output_fields should not include the vector field itself if not needed, can be large.
                        # If selectFields is empty or ['*'], fetch all non-vector fields.
                        actual_output_fields = []
                        if not selectFields or "*" in selectFields:
                            actual_output_fields = [f.name for f in collection.schema.fields if f.dtype != DataType.FLOAT_VECTOR]
                            if "id" not in actual_output_fields: # Ensure id is always there
                                actual_output_fields.append("id")
                        else:
                            actual_output_fields = list(set(selectFields + ["id"])) # Ensure id

                        search_results = collection.search(
                            data=[vector_to_search], # Expects list of query vectors
                            anns_field=vec_field_name,
                            param=search_params,
                            limit=limit + offset, # Fetch enough to cover offset and limit for manual slicing
                            expr=final_filter_expr,
                            output_fields=actual_output_fields,
                            consistency_level="Strong" # Or Bounded, Session, Eventually
                        )

                        # Apply offset manually after results are fetched
                        # Milvus search results are per query vector. We sent one.
                        hits = search_results[0]

                        # Manual slicing for offset
                        paginated_hits = hits[offset : offset + limit]

                        for hit in paginated_hits:
                            result_doc = hit.entity.to_dict() # Pymilvus Hit object has .entity
                            result_doc['score'] = hit.distance # Add score (distance)
                            # Remove vector field from result if it was accidentally included and large
                            # if vec_field_name in result_doc: del result_doc[vec_field_name]
                            results_list.append(result_doc)

                    elif isinstance(expr, MatchTextExpr):
                        # Placeholder for text search or hybrid search
                        # Milvus doesn't do traditional keyword text search on VARCHAR fields like ES.
                        # It can do it if scalar fields are indexed (e.g. with Trie for exact match on numbers/strings)
                        # or if using a future full-text search feature.
                        # A simple "like" or "==" might work for limited cases if field is indexed.
                        # For now, this part is not deeply implemented.
                        # We could try to add to `final_filter_expr` if it's a simple equality on an indexed scalar field.
                        logging.warning(f"MatchTextExpr not fully implemented for Milvus in this pass. Text field: {expr.field}, Text: {expr.text}")
                        # Example: if expr.field is indexed and we want exact match:
                        # text_filter = f"{expr.field} == '{expr.text}'"
                        # Then this would need to be combined with vector search (hybrid) or run as a separate query.
                        # This is complex. Raising NotImplemented for now.
                        # raise NotImplementedError("Milvus MatchTextExpr / Hybrid Search not implemented yet.")
                        pass # Skipping text search for now

                # TODO(developer): Handle orderBy if not just by vector distance.
                # Default order is by distance. Other sorting might require post-processing.

            except Exception as e:
                logging.error(f"Failed to search collection {collection_name} for tenant {self.tenant}: {e}")
                # Continue to next kb_id or raise

        # TODO(developer): If results_list came from multiple collections, might need global re-sorting/limiting.
        # For now, it's a simple concatenation.
        return results_list


    def get(self, chunkId: str, indexName: str, knowledgebaseIds: list[str]) -> dict | None:
        from pymilvus import Collection # DataType imported at top
        if not isinstance(knowledgebaseIds, list):
            knowledgebaseIds = [knowledgebaseIds]

        for kb_id in knowledgebaseIds:
            collection_name = f"{indexName}_{kb_id}".replace("-", "_")
            # Use consistent indexExist signature
            if not self.indexExist(indexName, kb_id):
                logging.debug(f"Collection {collection_name} does not exist for tenant {self.tenant}. Skipping for get operation.")
                continue

            try:
                collection = Collection(name=collection_name, using=f"default_{self.tenant}", db_name=self.db_name)
                collection.load() # Ensure collection is loaded

                expr = f"id == '{chunkId}'"
                # In Pymilvus 2.x, query() returns a list of entities (dictionaries).
                # output_fields=['*'] should work if schema is not overly complex with unsupported types for '*'
                # Or specify all known fields. For now, let's try with a limited set of common fields + dynamic.
                # Need to fetch all schema fields for output_fields=['*'] effectively
                # Use consistent get_schema_field_names signature
                schema_fields = self.get_schema_field_names(indexName, kb_id)
                if not schema_fields:
                    logging.error(f"Could not retrieve schema fields for {collection_name} via get_schema_field_names. Skipping get operation.")
                    continue

                results = collection.query(
                    expr=expr,
                    output_fields=schema_fields, # Fetch all schema-defined fields
                    limit=1 # We only expect one result for a given ID
                )

                if results:
                    # Results is a list of dicts. Get the first one.
                    # Vector field might be large, consider excluding it if not needed for 'get'
                    # For now, returning all fields.
                    doc = results[0]
                    # Remove the vector field from the result if it's too large and not needed for 'get'
                    # vector_field_to_remove = None
                    # for key, value in doc.items():
                    #     if isinstance(value, list) and len(value) > 0 and isinstance(value[0], float): # Heuristic for vector
                    #         vector_field_to_remove = key
                    #         break
                    # if vector_field_to_remove:
                    # del doc[vector_field_to_remove]
                    return doc
            except Exception as e:
                logging.error(f"Failed to get document with id {chunkId} from {collection_name} for tenant {self.tenant}: {e}")
                # Continue to check next knowledgebaseId

        return None

    def insert(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        from pymilvus import Collection # DataType imported at top
        collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")

        # Use consistent indexExist signature
        if not self.indexExist(indexName, knowledgebaseId):
            logging.error(f"Collection {collection_name} does not exist. Cannot insert data for tenant {self.tenant}.")
            raise ValueError(f"Collection {collection_name} does not exist.")

        try:
            collection = Collection(name=collection_name, using=f"default_{self.tenant}", db_name=self.db_name)
            collection.load() # Ensure collection is loaded

            # Determine vector field name from schema
            vector_field_name = None
            for field in collection.schema.fields:
                if field.dtype == DataType.FLOAT_VECTOR: # Corrected: field.dtype
                    vector_field_name = field.name
                    break
            if not vector_field_name:
                raise ValueError(f"No float vector field found in schema for collection {collection_name}")

            prepared_docs = []
            for doc in documents:
                # Ensure all schema fields are present or handle dynamic fields
                # For now, assume doc contains keys matching schema fields
                # and the vector data is under a common key like 'embedding' or must match vector_field_name

                # Create a copy to avoid modifying original document
                prepared_doc = copy.deepcopy(doc)

                # Handle the vector field:
                # Option 1: Assume 'embedding' key in input doc holds the vector
                if 'embedding' in prepared_doc and vector_field_name not in prepared_doc:
                    prepared_doc[vector_field_name] = prepared_doc.pop('embedding')
                elif vector_field_name not in prepared_doc:
                    # Option 2: Or if the vector field is directly named, e.g. q_1024_vec
                    # This case is implicitly handled if prepared_doc already has vector_field_name
                    # If neither 'embedding' nor the actual vector_field_name is present, it's an error
                     raise ValueError(f"Vector data not found in document under key 'embedding' or '{vector_field_name}' for doc ID {doc.get('id')}")


                # Ensure all defined schema fields exist in the doc, or Milvus might error
                # if not using dynamic fields strictly. With dynamic fields, it's more flexible.
                # For now, assume documents are well-formed for the existing schema fields.
                # Example: ensure kb_id is set if not already in doc
                if 'kb_id' not in prepared_doc and knowledgebaseId:
                     prepared_doc['kb_id'] = knowledgebaseId

                prepared_docs.append(prepared_doc)

            if not prepared_docs:
                return []

            # Milvus insert expects list of dicts if dynamic_field is enabled and data matches field names
            # Or list of lists (column-oriented)
            # Using list of dicts here as it's more convenient and schema has enable_dynamic_field=True
            mutation_result = collection.insert(prepared_docs)
            collection.flush() # Ensure data is written to disk

            # mutation_result contains primary keys of inserted entities
            # Check for errors if possible, though pymilvus usually raises exceptions on failure.
            # For now, assume success if no exception.
            # If there were row-level errors, mutation_result.err_count might be useful in some contexts.
            if hasattr(mutation_result, 'err_count') and mutation_result.err_count > 0 : # Check if err_count attr exists
                 # This part of API might vary; consult specific Pymilvus version for error details
                 logging.error(f"Milvus insert operation reported {mutation_result.err_count} errors for collection {collection_name}.")
                 # How to get specific error messages per row is not straightforward from mutation_result alone.
                 # This would typically require pre-validation or handling exceptions.
                 # For now, returning a generic error message.
                 # It's better to raise an exception if err_count > 0
                 raise Exception(f"Milvus insert operation reported {mutation_result.err_count} errors.")


            logging.info(f"Successfully inserted {len(mutation_result.primary_keys)} documents into {collection_name} for tenant {self.tenant}.")
            # Return empty list on success as per review decision
            return []

        except Exception as e:
            logging.error(f"Failed to insert data into collection {collection_name} for tenant {self.tenant}: {e}")
            raise # Re-raise the exception for the caller to handle

    def update(self, condition: dict, newValue: dict, indexName: str, knowledgebaseId: str) -> bool:
        from pymilvus import Collection # DataType imported at top
        collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")

        # Use consistent indexExist signature
        if not self.indexExist(indexName, knowledgebaseId):
            logging.warning(f"Collection {collection_name} does not exist. Cannot update data for tenant {self.tenant}.")
            return False

        if "id" not in condition or not isinstance(condition["id"], str):
            logging.error("Update condition must contain a string 'id' of the document to update.")
            return False

        doc_id_to_update = condition["id"]

        try:
            collection = Collection(name=collection_name, using=f"default_{self.tenant}", db_name=self.db_name)
            collection.load()

            # 1. Fetch the existing entity
            # Need all schema fields to reconstruct the document for re-insertion
            schema_fields = [field.name for field in collection.schema.fields]
            query_expr = f"id == '{doc_id_to_update}'"

            # Add kb_id to query if it was part of the condition, for extra safety, though id should be unique.
            if "kb_id" in condition and isinstance(condition["kb_id"], str) :
                 query_expr += f" and kb_id == '{condition['kb_id']}'"

            existing_entities = collection.query(expr=query_expr, output_fields=schema_fields, limit=1)

            if not existing_entities:
                logging.warning(f"Document with id '{doc_id_to_update}' not found in {collection_name} for tenant {self.tenant}. Cannot update.")
                return False

            original_doc = existing_entities[0]

            # 2. Delete the old entity (critical step, data loss if insert fails)
            # Use a very specific delete expression to avoid accidental multi-deletes.
            delete_expr = f"id == '{doc_id_to_update}'"
            if "kb_id" in condition and isinstance(condition["kb_id"], str) : # Ensure we only delete from the correct kb if specified
                delete_expr += f" and kb_id == '{condition['kb_id']}'"

            del_result = collection.delete(delete_expr)
            if del_result.delete_count == 0:
                logging.warning(f"No document found to delete with id '{doc_id_to_update}' during update operation, though it was fetched. This is unexpected.")
                # Proceeding to insert might lead to duplicates if there's a race condition or eventual consistency issue.
                # For now, we'll proceed with insert as fetch succeeded.

            # collection.flush() # Flush delete operation immediately if needed, though can be batched with insert flush

            # 3. Prepare the new document
            updated_doc = copy.deepcopy(original_doc)

            # Determine vector field name from schema to handle it if present in newValue
            vector_field_name = None
            for field in collection.schema.fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    vector_field_name = field.name
                    break

            for key, value in newValue.items():
                if key == "id" and value != doc_id_to_update:
                    logging.warning(f"Attempt to change 'id' during update for doc {doc_id_to_update} is not allowed. Skipping 'id' field.")
                    continue
                if key == vector_field_name and 'embedding' in newValue : # If 'embedding' is passed in newValue, it's the new vector
                    updated_doc[vector_field_name] = newValue['embedding']
                elif key == 'embedding' and vector_field_name : # if newValue has 'embedding' key, map it to actual vector field
                    updated_doc[vector_field_name] = value
                else:
                    updated_doc[key] = value

            # Ensure kb_id from original doc or path is preserved if not in newValue
            if 'kb_id' not in updated_doc:
                updated_doc['kb_id'] = original_doc.get('kb_id', knowledgebaseId)


            # 4. Insert the new document
            # The insert method expects a list of documents
            insert_result = collection.insert([updated_doc])
            collection.flush() # Flush both delete and insert

            if hasattr(insert_result, 'err_count') and insert_result.err_count > 0:
                logging.error(f"Failed to insert updated document for id '{doc_id_to_update}' in {collection_name}. Errors: {insert_result.err_count}. Data may have been deleted but not replaced.")
                # Attempt to rollback delete? Complex. For now, report error.
                # This is a critical failure state.
                return False # Or raise an exception

            logging.info(f"Document with id '{doc_id_to_update}' updated successfully in {collection_name} for tenant {self.tenant}.")
            return True

        except Exception as e:
            logging.error(f"Failed to update document with id '{doc_id_to_update}' in collection {collection_name} for tenant {self.tenant}: {e}")
            # Potentially try to re-insert original if delete succeeded but insert failed? Very complex recovery.
            return False # Or raise

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        from pymilvus import Collection # DataType imported at top
        collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")

        # Use consistent indexExist signature
        if not self.indexExist(indexName, knowledgebaseId):
            logging.warning(f"Collection {collection_name} does not exist. Cannot delete data for tenant {self.tenant}.")
            return 0

        try:
            collection = Collection(name=collection_name, using=f"default_{self.tenant}", db_name=self.db_name)
            collection.load()

            expr_parts = []
            if "id" in condition:
                ids = condition["id"]
                if isinstance(ids, list):
                    if not ids: return 0 # Nothing to delete
                    # Ensure all IDs are strings and properly quoted
                    formatted_ids = [f"'{str(i)}'" if isinstance(i, str) else str(i) for i in ids]
                    expr_parts.append(f"id in [{', '.join(formatted_ids)}]")
                elif isinstance(ids, str):
                    expr_parts.append(f"id == '{ids}'")
                else:
                    logging.error(f"Unsupported type for 'id' in delete condition: {type(ids)}")
                    return 0

            if "kb_id" in condition: # This might be redundant if kb_id is part of collection name, but can act as safeguard
                kb_filter_id = condition["kb_id"]
                if isinstance(kb_filter_id, str):
                    expr_parts.append(f"kb_id == '{kb_filter_id}'")
                else:
                    logging.error(f"Unsupported type for 'kb_id' in delete condition: {type(kb_filter_id)}")
                    # Potentially skip this filter or return error

            if not expr_parts:
                logging.warning("Delete condition is empty. No documents will be deleted.")
                return 0

            expr = " and ".join(expr_parts)

            logging.debug(f"Attempting to delete from {collection_name} with expression: {expr}")
            mutation_result = collection.delete(expr)
            collection.flush()

            deleted_count = mutation_result.delete_count
            logging.info(f"Successfully deleted {deleted_count} documents from {collection_name} for tenant {self.tenant} with expression: {expr}.")
            return deleted_count

        except Exception as e:
            logging.error(f"Failed to delete data from collection {collection_name} for tenant {self.tenant}: {e}")
            raise # Re-raise for caller to handle

    def getTotal(self, name: str, knowledgebaseId: str = None) -> int:
        """
        Gets the total number of entities in a collection.
        'name' is the base index name.
        'knowledgebaseId' is the specific KB ID.
        If 'knowledgebaseId' is None, 'name' is assumed to be the full collection name.
        """
        from pymilvus import Collection
        if knowledgebaseId:
            collection_name = f"{name}_{knowledgebaseId}".replace("-", "_")
        else:
            collection_name = name # Assume name is the full collection name

        try:
            # Use consistent indexExist signature
            if not self.indexExist(name, knowledgebaseId if knowledgebaseId else None):
                logging.warning(f"Collection {collection_name} does not exist for tenant {self.tenant}. Cannot get total count.")
                return 0

            collection = Collection(name=collection_name, using=f"default_{self.tenant}", db_name=self.db_name)
            # For num_entities, loading is not strictly required, but flushing ensures accuracy after recent operations.
            collection.flush()
            return collection.num_entities
        except Exception as e:
            logging.error(f"Failed to get total count for collection {collection_name} for tenant {self.tenant}: {e}")
            return 0

    def getChunkIds(self, name: str, knowledgebaseId: str = None) -> list[str]:
        from pymilvus import Collection
        # The 'name' parameter here is likely indexName.
        # Construct collection_name similar to other methods.
        # If knowledgebaseId is not given, this method might need to iterate or have a convention.
        # For now, assume knowledgebaseId is provided or the 'name' is the full collection_name.

        # Clarification: Based on DocStoreConnection, 'name' is the collection/index name.
        # The prompt for this subtask didn't specify knowledgebaseId for getChunkIds,
        # but other methods like insert/delete/get use indexName + knowledgebaseId.
        # Let's assume 'name' is the base indexName and knowledgebaseId is also available,
        # or 'name' is the already combined collection name.
        # For consistency with other new methods, I'll assume 'name' is indexName and knowledgebaseId is separate.

        # If knowledgebaseId is None, we might be operating on a raw collection name 'name'
        # or this indicates an issue with how it's called.
        # Let's assume if kb_id is None, 'name' is the full collection name.
        if knowledgebaseId:
            collection_name = f"{name}_{knowledgebaseId}".replace("-", "_")
        else:
            # This case needs clarification on how full collection names are passed if not via indexName+kb_id
            # For now, assume 'name' can be a full collection name if kb_id is not specified.
            # However, this is inconsistent with createIdx, insert, delete, update, get.
            # Defaulting to the pattern: indexName + knowledgebaseId.
            # If this method is called with only 'name', it should be the full collection name.
            # To align with other methods, let's adjust the thinking or expect both.
            # The original signature was getChunkIds(self, name:str). I'll stick to that and assume 'name' is full collection name.
            collection_name = name # Assuming 'name' is the full collection_name if knowledgebaseId is None

        # Use consistent indexExist signature
        if not self.indexExist(name, knowledgebaseId if knowledgebaseId else None):
            logging.warning(f"Collection {collection_name} does not exist. Cannot get chunk IDs for tenant {self.tenant}.")
            return []

        try:
            collection = Collection(name=collection_name, using=f"default_{self.tenant}", db_name=self.db_name)
            collection.load()

            # Query for all IDs. Milvus's query capabilities for "all" can be done by empty expr or "id != ''"
            # and iterating with limit/offset if collection is huge.
            # For now, let's try to get all IDs, assuming not excessively many.
            # Milvus does not directly support `select id from collection` without a search/query condition.
            # We can use a query with a general condition like "id != ''" or "pk_field != ''"
            # and specify output_fields=['id'].
            # The primary key field is 'id'.

            # Need to handle potential large number of IDs. Milvus query has limits.
            # Iterative query:
            all_ids = []
            offset = 0
            # Milvus default query limit is 16384. Let's use a smaller batch size for iteration.
            batch_size = 10000

            while True:
                results = collection.query(
                    expr="id != ''", # A general expression to get all entities
                    output_fields=["id"],
                    limit=batch_size,
                    offset=offset
                )
                if not results:
                    break
                all_ids.extend([res["id"] for res in results])
                if len(results) < batch_size: # Last page
                    break
                offset += len(results)

            return all_ids
        except Exception as e:
            logging.error(f"Failed to get chunk IDs from collection {collection_name} for tenant {self.tenant}: {e}")
            return []

    # Renamed from getFields to avoid conflict with the result-processing getFields method below
    def get_schema_field_names(self, collection_name: str, knowledgebaseId: str = None) -> list[str]:
        """
        Gets the field names from the schema of a collection.
        'collection_name' is the base index name.
        'knowledgebaseId' is the specific KB ID.
        If 'knowledgebaseId' is None, 'collection_name' is assumed to be the full collection name.
        """
        from pymilvus import Collection
        if knowledgebaseId:
            full_collection_name = f"{collection_name}_{knowledgebaseId}".replace("-", "_")
        else:
            full_collection_name = collection_name # if knowledgebaseId is None, collection_name is 'collection_name' (which is 'name' from params)

        try:
            # Use consistent indexExist signature
            if not self.indexExist(collection_name, knowledgebaseId if knowledgebaseId else None): # Pass base name and kb_id
                logging.warning(f"Collection {full_collection_name} does not exist for tenant {self.tenant} when trying to get schema fields.")
                return []
            collection = Collection(name=full_collection_name, using=f"default_{self.tenant}", db_name=self.db_name)
            # No need to load collection for schema access
            schema = collection.schema
            return [field.name for field in schema.fields]
        except Exception as e:
            logging.error(f"Failed to get schema fields for collection {full_collection_name} for tenant {self.tenant}: {e}")
            return []

    def getFields(self, res: list[dict], fields: list[str]) -> dict[str, dict]:
        """
        Extracts specified fields from a list of search result documents.
        'res': A list of dictionaries, where each dictionary is a hit from Milvus search (should include 'id' and other data fields).
        'fields': A list of field names to extract.
        Returns: A dictionary where keys are document IDs and values are dictionaries of the extracted fields.
        """
        res_fields = {}
        if not fields:
            return {}

        for doc in res:
            if 'id' not in doc:
                logging.warning("Document in search results missing 'id' field. Skipping.")
                continue

            doc_id = doc['id']
            extracted_data = {}
            for field_name in fields:
                if field_name in doc:
                    extracted_data[field_name] = doc[field_name]

            if extracted_data: # Only add if there are some fields extracted
                res_fields[doc_id] = extracted_data
        return res_fields

    def getHighlight(self, res, keywords: list[str], fieldnm: str) -> dict[str, str]:
        """
        Milvus does not provide built-in highlighting.
        This method returns an empty dictionary, consistent with other connectors
        that do not support highlighting directly.
        'res': The search result list of dicts (unused in this implementation).
        'keywords': List of keywords to highlight (unused).
        'fieldnm': Field name where highlighting would be applied (unused).
        """
        # logging.debug("MilvusConnection.getHighlight called, returning empty dict as Milvus does not support it.")
        return {}

    def getAggregation(self, res, fieldnm: str) -> list[tuple[str, int]]:
        """
        Milvus does not offer direct aggregation capabilities similar to Elasticsearch's terms aggregation.
        This method returns an empty list.
        'res': The search result list of dicts (unused in this implementation).
        'fieldnm': Field name for aggregation (unused).
        """
        # logging.debug("MilvusConnection.getAggregation called, returning empty list as Milvus does not support it directly.")
        return []

    def sql(self, sql: str, fetch_size: int = -1, format: str = "dataframe") -> pd.DataFrame | list[dict]: # Cosmetic: space before colon
        """
        Milvus does not support SQL queries directly in the same way as an SQL database or Elasticsearch SQL.
        Use specific search/query methods instead.
        """
        logging.warning("Milvus does not support SQL queries directly. Use search/query methods.")
        raise NotImplementedError("Milvus does not support SQL queries directly. Use search/query methods.")
