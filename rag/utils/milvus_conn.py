import copy
import json
import logging
import os
import re
import time

from pymilvus import DataType, Collection, CollectionSchema, FieldSchema, connections, utility
from rag import settings
from rag.utils import singleton
from api.utils.file_utils import get_project_base_directory
from rag.utils.doc_store_conn import (
    DocStoreConnection,
    MatchExpr,
    MatchTextExpr,
    MatchDenseExpr,
    OrderByExpr,
)

ATTEMPT_TIME = 2

logger = logging.getLogger('ragflow.milvus_conn')


def field_keyword(field_name: str):
    """Check if field is a keyword field based on naming convention"""
    if field_name == "source_id" or (field_name.endswith("_kwd") and field_name != "docnm_kwd" and field_name != "knowledge_graph_kwd"):
        return True
    return False


def get_milvus_field_type(field_name: str, field_info: dict) -> tuple[DataType, dict]:
    """Convert infinity mapping field type to Milvus field type"""
    field_type = field_info.get("type", "varchar")
    default_value = field_info.get("default", "")

    # Handle vector fields
    if "_vec" in field_name:
        # Extract dimension from field name like q_1024_vec
        match = re.search(r'q_(\d+)_vec', field_name)
        if match:
            dim = int(match.group(1))
            return DataType.FLOAT_VECTOR, {"dim": dim}

    # Handle other field types
    if field_type == "integer":
        return DataType.INT32, {}
    elif field_type == "float":
        return DataType.FLOAT, {}
    elif field_type == "varchar":
        # Determine max length based on field usage
        if field_name in ["id", "doc_id", "kb_id", "img_id"]:
            max_length = 255
        elif field_name in ["content_ltks", "content_sm_ltks"]:
            max_length = 65535
        else:
            max_length = 1024
        return DataType.VARCHAR, {"max_length": max_length}
    else:
        # Default to varchar
        return DataType.VARCHAR, {"max_length": 1024}


def get_milvus_field_type_from_config(field_name: str, field_info: dict) -> tuple[DataType, dict]:
    """Convert milvus mapping config field type to Milvus field type"""
    field_type = field_info.get("type", "varchar")
    params = {}

    # Handle different field types
    if field_type == "varchar":
        params["max_length"] = field_info.get("max_length", 65535)
        if field_info.get("is_primary", False):
            params["is_primary"] = True
            params["auto_id"] = field_info.get("auto_id", False)
        return DataType.VARCHAR, params
    elif field_type == "int64":
        return DataType.INT64, params
    elif field_type == "float":
        return DataType.FLOAT, params
    elif field_type == "float_vector":
        params["dim"] = field_info.get("dim", 1024)
        return DataType.FLOAT_VECTOR, params
    else:
        # Default to varchar for unknown types
        params["max_length"] = field_info.get("max_length", 65535)
        return DataType.VARCHAR, params


@singleton
class MilvusConnection(DocStoreConnection):
    def __init__(self):
        self.alias = "default_milvus"
        milvus_config = settings.MILVUS or {}
        logger.info(f"Use Milvus {milvus_config.get('host', 'localhost')}:{milvus_config.get('port', '19530')} as the doc engine.")

        for _ in range(ATTEMPT_TIME * 12):  # More attempts for connection
            try:
                self._create_connection()
                if self._test_connection():
                    break
                logger.warning(f"Milvus connection test failed. Retrying...")
                time.sleep(5)
            except Exception as e:
                logger.warning(f"{str(e)}. Waiting Milvus to be healthy.")
                time.sleep(5)
        else:
            msg = f"Milvus {milvus_config.get('host', 'localhost')}:{milvus_config.get('port', '19530')} is unhealthy in 120s."
            logger.error(msg)
            raise Exception(msg)

        # Load field mapping
        fp_mapping = os.path.join(get_project_base_directory(), "conf", "milvus_mapping.json")
        if not os.path.exists(fp_mapping):
            msg = f"Milvus mapping file not found at {fp_mapping}"
            logger.error(msg)
            raise Exception(msg)
        self.mapping_config = json.load(open(fp_mapping, "r"))
        self.mapping = self.mapping_config["mappings"]["properties"]
        logger.info(f"Milvus connection established successfully.")

    def _create_connection(self):
        """Create connection to Milvus"""
        milvus_config = settings.MILVUS or {}
        host = milvus_config.get("host", "localhost")
        port = milvus_config.get("port", "19530")
        user = milvus_config.get("user", "")
        password = milvus_config.get("password", "")

        conn_params = {
            "alias": self.alias,
            "host": host,
            "port": port,
        }

        if user and password:
            conn_params["user"] = user
            conn_params["password"] = password

        connections.connect(**conn_params)
        logger.info(f"Connected to Milvus at {host}:{port}")

    def _test_connection(self) -> bool:
        """Test if connection is healthy"""
        try:
            # Try to list collections as a health check
            utility.list_collections(using=self.alias)
            return True
        except Exception as e:
            logger.warning(f"Milvus connection test failed: {e}")
            return False

    """
    Database operations
    """

    def dbType(self) -> str:
        return "milvus"

    def health(self) -> dict:
        try:
            collections = utility.list_collections(using=self.alias)
            return {
                "type": "milvus",
                "status": "green",
                "collections": len(collections)
            }
        except Exception as e:
            return {
                "type": "milvus",
                "status": "red",
                "error": str(e)
            }

    """
    Table operations
    """

    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):
        collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")

        if self.indexExist(indexName, knowledgebaseId):
            logger.info(f"Collection {collection_name} already exists. Skipping creation.")
            return True

        try:
            # Create schema based on mapping
            fields = []
            vector_field_name = f"q_{vectorSize}_vec"

            # Add fields from mapping properties
            for field_name, field_info in self.mapping.items():
                if field_name == vector_field_name or "_vec" in field_name:
                    continue  # Will add vector field separately

                dtype, params = get_milvus_field_type_from_config(field_name, field_info)
                field_schema = FieldSchema(
                    name=field_name,
                    dtype=dtype,
                    description=field_info.get("description", f"Field {field_name}"),
                    **params
                )
                fields.append(field_schema)

            # Add vector field
            vector_fields = self.mapping_config["mappings"].get("vector_fields", {})
            if vector_field_name in vector_fields:
                vector_info = vector_fields[vector_field_name]
                fields.append(FieldSchema(
                    name=vector_field_name,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=vector_info["dim"],
                    description=vector_info.get("description", "Query vector embedding")
                ))
            else:
                # Fallback to dynamic vector field creation
                fields.append(FieldSchema(
                    name=vector_field_name,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=vectorSize,
                    description="Query vector embedding"
                ))

            schema = CollectionSchema(
                fields=fields,
                description=f"Collection for {indexName} of KB {knowledgebaseId}",
                enable_dynamic_field=True
            )

            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.alias
            )

            # Create index for vector field using config
            index_config = self.mapping_config.get("settings", {}).get("index", {})
            index_params = {
                "metric_type": index_config.get("metric_type", "L2"),
                "index_type": index_config.get("index_type", "IVF_FLAT"),
                "params": index_config.get("params", {"nlist": 128}),
            }

            collection.create_index(field_name=vector_field_name, index_params=index_params)
            collection.load()

            logger.info(f"Collection {collection_name} created and loaded successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to create index {collection_name}: {e}")
            raise

    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")
        try:
            if self.indexExist(indexName, knowledgebaseId):
                utility.drop_collection(collection_name, using=self.alias)
                logger.info(f"Collection {collection_name} deleted successfully.")
            else:
                logger.warning(f"Collection {collection_name} does not exist, cannot delete.")
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise

    def indexExist(self, indexName: str, knowledgebaseId: str | None = None) -> bool:
        if knowledgebaseId:
            collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")
        else:
            collection_name = indexName

        try:
            return utility.has_collection(collection_name, using=self.alias)
        except Exception as e:
            logger.error(f"Failed to check if index {collection_name} exists: {e}")
            return False

    """
    CRUD operations
    """

    def search(
        self,
        selectFields: list[str],
        highlightFields: list[str],
        condition: dict,
        matchExprs: list[MatchExpr],
        orderBy: OrderByExpr,
        offset: int,
        limit: int,
        indexNames: str | list[str],
        knowledgebaseIds: list[str],
        aggFields: list[str] = [],
        rank_feature: dict | None = None
    ):
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")
        assert isinstance(indexNames, list) and len(indexNames) > 0

        if not knowledgebaseIds:
            logger.warning("Search called with no knowledgebaseIds.")
            return {"hits": {"hits": [], "total": {"value": 0}}}

        results_list = []
        total_hits = 0

        # Process each index and knowledge base combination
        for indexName in indexNames:
            for kb_id in knowledgebaseIds:
                collection_name = f"{indexName}_{kb_id}".replace("-", "_")

                if not self.indexExist(indexName, kb_id):
                    logger.warning(f"Collection {collection_name} does not exist. Skipping search for this KB.")
                    continue

                try:
                    collection = Collection(name=collection_name, using=self.alias)
                    collection.load()

                    # Build filter expression from condition
                    filter_expr_parts = []
                    if condition:
                        for field, value in condition.items():
                            if field == "kb_id":
                                continue  # Skip kb_id as it's implicit in collection name
                            if isinstance(value, list):
                                formatted_values = [f"'{v}'" if isinstance(v, str) else str(v) for v in value]
                                filter_expr_parts.append(f"{field} in [{', '.join(formatted_values)}]")
                            elif isinstance(value, str):
                                filter_expr_parts.append(f"{field} == '{value}'")
                            else:
                                filter_expr_parts.append(f"{field} == {value}")

                    final_filter_expr = " and ".join(filter_expr_parts) if filter_expr_parts else ""
                    logger.debug(f"Milvus search filter expression for {collection_name}: '{final_filter_expr}'")

                    # Process match expressions
                    for expr in matchExprs:
                        if isinstance(expr, MatchDenseExpr):
                            # Vector search
                            vector_field_name = expr.vector_column_name
                            if not vector_field_name:
                                # Find vector field from schema
                                for field in collection.schema.fields:
                                    if field.dtype == DataType.FLOAT_VECTOR:
                                        vector_field_name = field.name
                                        break

                            if not vector_field_name:
                                logger.error(f"No vector field found in collection {collection_name}")
                                continue

                            # Prepare search parameters
                            milvus_config = settings.MILVUS or {}
                            search_config = milvus_config.get("search", {})
                            search_params = {
                                "metric_type": search_config.get("metric_type", "L2"),
                                "params": search_config.get("params", {"nprobe": 10}),
                            }

                            # Determine output fields
                            output_fields = []
                            if not selectFields or "*" in selectFields:
                                output_fields = [f.name for f in collection.schema.fields if f.dtype != DataType.FLOAT_VECTOR]
                            else:
                                output_fields = list(set(selectFields + ["id"]))

                            # Perform vector search
                            search_results = collection.search(
                                data=[expr.embedding_data],
                                anns_field=vector_field_name,
                                param=search_params,
                                limit=limit + offset,
                                expr=final_filter_expr if final_filter_expr else None,
                                output_fields=output_fields,
                                consistency_level=search_config.get("consistency_level", "Strong")
                            )

                            # Process results
                            if search_results and len(search_results) > 0:
                                hits = search_results[0]
                                paginated_hits = hits[offset:offset + limit]

                                for hit in paginated_hits:
                                    result_doc = hit.entity.to_dict()
                                    result_doc['_score'] = hit.distance
                                    results_list.append(result_doc)

                                total_hits += len(hits)

                        elif isinstance(expr, MatchTextExpr):
                            # Text search not directly supported in Milvus
                            # Could be implemented with scalar field filtering
                            logger.warning("Text search not implemented for Milvus")
                            pass

                except Exception as e:
                    logger.error(f"Failed to search collection {collection_name}: {e}")
                    continue

        # Format results to match expected format
        return {
            "hits": {
                "hits": [{"_source": doc, "_id": doc.get("id"), "_score": doc.get("_score", 0)} for doc in results_list],
                "total": {"value": total_hits}
            }
        }


    def get(self, chunkId: str, indexName: str, knowledgebaseIds: list[str]) -> dict | None:
        if not isinstance(knowledgebaseIds, list):
            knowledgebaseIds = [knowledgebaseIds]

        for kb_id in knowledgebaseIds:
            collection_name = f"{indexName}_{kb_id}".replace("-", "_")

            if not self.indexExist(indexName, kb_id):
                logger.debug(f"Collection {collection_name} does not exist. Skipping for get operation.")
                continue

            try:
                collection = Collection(name=collection_name, using=self.alias)
                collection.load()

                expr = f"id == '{chunkId}'"

                # Get all non-vector fields for output
                output_fields = []
                for field in collection.schema.fields:
                    if field.dtype != DataType.FLOAT_VECTOR:
                        output_fields.append(field.name)

                results = collection.query(
                    expr=expr,
                    output_fields=output_fields,
                    limit=1
                )

                if results:
                    doc = results[0]
                    doc["id"] = chunkId  # Ensure id is present
                    return doc

            except Exception as e:
                logger.error(f"Failed to get document with id {chunkId} from {collection_name}: {e}")
                continue

        return None

    def insert(self, rows: list[dict], indexName: str, knowledgebaseId: str | None = None) -> list[str]:
        if knowledgebaseId is None:
            raise ValueError("knowledgebaseId cannot be None")

        collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")

        if not self.indexExist(indexName, knowledgebaseId):
            # Auto-create collection if it doesn't exist
            if rows:
                # Try to infer vector size from first document
                vector_size = 0
                for key, value in rows[0].items():
                    if "_vec" in key and isinstance(value, list):
                        vector_size = len(value)
                        break
                if vector_size > 0:
                    self.createIdx(indexName, knowledgebaseId, vector_size)
                else:
                    raise ValueError(f"Collection {collection_name} does not exist and cannot infer vector size.")
            else:
                raise ValueError(f"Collection {collection_name} does not exist.")

        try:
            collection = Collection(name=collection_name, using=self.alias)
            collection.load()

            # Find vector field name from schema
            vector_field_name = None
            for field in collection.schema.fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    vector_field_name = field.name
                    break

            prepared_docs = []
            for doc in rows:
                prepared_doc = copy.deepcopy(doc)

                # Handle vector field mapping
                if vector_field_name:
                    if 'embedding' in prepared_doc and vector_field_name not in prepared_doc:
                        prepared_doc[vector_field_name] = prepared_doc.pop('embedding')
                    elif vector_field_name not in prepared_doc:
                        # Look for any vector field in the document
                        for key, value in prepared_doc.items():
                            if "_vec" in key and isinstance(value, list):
                                prepared_doc[vector_field_name] = prepared_doc.pop(key)
                                break

                # Ensure kb_id is set
                if 'kb_id' not in prepared_doc and knowledgebaseId:
                    prepared_doc['kb_id'] = knowledgebaseId

                # Ensure all required fields from mapping are present with default values
                for field_name, field_info in self.mapping.items():
                    if field_name not in prepared_doc:
                        default_value = field_info.get("default", "")
                        # Skip vector fields as they are handled separately
                        if "_vec" in field_name:
                            continue
                        prepared_doc[field_name] = default_value

                # Handle special field transformations similar to infinity_conn
                for key, value in list(prepared_doc.items()):
                    if field_keyword(key):
                        if isinstance(value, list):
                            prepared_doc[key] = "###".join(value)
                    elif re.search(r"_feas$", key):
                        prepared_doc[key] = json.dumps(value)
                    elif key == "position_int":
                        if isinstance(value, list):
                            arr = [num for row in value for num in row]
                            prepared_doc[key] = "_".join(f"{num:08x}" for num in arr)
                    elif key in ["page_num_int", "top_int"]:
                        if isinstance(value, list):
                            prepared_doc[key] = "_".join(f"{num:08x}" for num in value)

                prepared_docs.append(prepared_doc)

            if not prepared_docs:
                return []

            # Delete existing documents with same IDs first
            ids_to_delete = [doc["id"] for doc in prepared_docs if "id" in doc]
            if ids_to_delete:
                try:
                    formatted_ids = [f"'{id}'" for id in ids_to_delete]
                    delete_expr = f"id in [{', '.join(formatted_ids)}]"
                    collection.delete(delete_expr)
                except Exception as e:
                    logger.warning(f"Failed to delete existing documents: {e}")

            mutation_result = collection.insert(prepared_docs)
            collection.flush()

            logger.info(f"Successfully inserted {len(prepared_docs)} documents into {collection_name}.")
            return []

        except Exception as e:
            logger.error(f"Failed to insert data into collection {collection_name}: {e}")
            raise

    def update(self, condition: dict, newValue: dict, indexName: str, knowledgebaseId: str) -> bool:
        collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")

        if not self.indexExist(indexName, knowledgebaseId):
            logger.warning(f"Collection {collection_name} does not exist. Cannot update data.")
            return False

        if "id" not in condition or not isinstance(condition["id"], str):
            logger.error("Update condition must contain a string 'id' of the document to update.")
            return False

        doc_id_to_update = condition["id"]

        try:
            collection = Collection(name=collection_name, using=self.alias)
            collection.load()

            # Fetch existing document
            query_expr = f"id == '{doc_id_to_update}'"
            output_fields = [field.name for field in collection.schema.fields if field.dtype != DataType.FLOAT_VECTOR]

            existing_entities = collection.query(expr=query_expr, output_fields=output_fields, limit=1)

            if not existing_entities:
                logger.warning(f"Document with id '{doc_id_to_update}' not found in {collection_name}. Cannot update.")
                return False

            original_doc = existing_entities[0]

            # Delete old document
            delete_expr = f"id == '{doc_id_to_update}'"
            collection.delete(delete_expr)

            # Prepare updated document
            updated_doc = copy.deepcopy(original_doc)

            # Find vector field name
            vector_field_name = None
            for field in collection.schema.fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    vector_field_name = field.name
                    break

            # Apply updates with field transformations
            for key, value in newValue.items():
                if key == "id" and value != doc_id_to_update:
                    logger.warning(f"Attempt to change 'id' during update is not allowed. Skipping 'id' field.")
                    continue

                if key == 'embedding' and vector_field_name:
                    updated_doc[vector_field_name] = value
                elif field_keyword(key):
                    if isinstance(value, list):
                        updated_doc[key] = "###".join(value)
                    else:
                        updated_doc[key] = value
                elif re.search(r"_feas$", key):
                    updated_doc[key] = json.dumps(value)
                elif key == "position_int":
                    if isinstance(value, list):
                        arr = [num for row in value for num in row]
                        updated_doc[key] = "_".join(f"{num:08x}" for num in arr)
                elif key in ["page_num_int", "top_int"]:
                    if isinstance(value, list):
                        updated_doc[key] = "_".join(f"{num:08x}" for num in value)
                else:
                    updated_doc[key] = value

            # Ensure kb_id is preserved
            if 'kb_id' not in updated_doc:
                updated_doc['kb_id'] = original_doc.get('kb_id', knowledgebaseId)

            # Ensure all required fields from mapping are present with default values
            for field_name, field_info in self.mapping.items():
                if field_name not in updated_doc:
                    default_value = field_info.get("default", "")
                    # Skip vector fields as they are handled separately
                    if "_vec" in field_name:
                        continue
                    updated_doc[field_name] = default_value

            # Insert updated document
            collection.insert([updated_doc])
            collection.flush()

            logger.info(f"Document with id '{doc_id_to_update}' updated successfully in {collection_name}.")
            return True

        except Exception as e:
            logger.error(f"Failed to update document with id '{doc_id_to_update}' in collection {collection_name}: {e}")
            return False

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")

        if not self.indexExist(indexName, knowledgebaseId):
            logger.warning(f"Collection {collection_name} does not exist. Cannot delete data.")
            return 0

        try:
            collection = Collection(name=collection_name, using=self.alias)
            collection.load()

            expr_parts = []
            if "id" in condition:
                ids = condition["id"]
                if isinstance(ids, list):
                    if not ids:
                        return 0
                    formatted_ids = [f"'{str(i)}'" for i in ids]
                    expr_parts.append(f"id in [{', '.join(formatted_ids)}]")
                elif isinstance(ids, str):
                    expr_parts.append(f"id == '{ids}'")
                else:
                    logger.error(f"Unsupported type for 'id' in delete condition: {type(ids)}")
                    return 0

            # Add other condition filters
            for key, value in condition.items():
                if key in ["id", "kb_id"]:
                    continue
                if isinstance(value, str):
                    expr_parts.append(f"{key} == '{value}'")
                elif isinstance(value, (int, float)):
                    expr_parts.append(f"{key} == {value}")

            if not expr_parts:
                logger.warning("Delete condition is empty. No documents will be deleted.")
                return 0

            expr = " and ".join(expr_parts)

            logger.debug(f"Attempting to delete from {collection_name} with expression: {expr}")
            mutation_result = collection.delete(expr)
            collection.flush()

            deleted_count = mutation_result.delete_count
            logger.info(f"Successfully deleted {deleted_count} documents from {collection_name} with expression: {expr}.")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete data from collection {collection_name}: {e}")
            raise

    """
    Helper functions for search result
    """

    def getTotal(self, res):
        if isinstance(res, dict) and "hits" in res and "total" in res["hits"]:
            return res["hits"]["total"]["value"]
        return 0

    def getChunkIds(self, res):
        if isinstance(res, dict) and "hits" in res:
            return [hit["_id"] for hit in res["hits"]["hits"]]
        return []

    def getFields(self, res, fields: list[str]) -> dict[str, dict]:
        """Extract specified fields from search results"""
        res_fields = {}
        if not fields:
            return {}

        # Handle different result formats
        hits = []
        if isinstance(res, dict) and "hits" in res:
            hits = res["hits"]["hits"]
        elif isinstance(res, list):
            hits = res

        for hit in hits:
            doc = hit.get("_source", hit)
            doc_id = doc.get("id") or hit.get("_id")

            if not doc_id:
                logger.warning("Document in search results missing 'id' field. Skipping.")
                continue

            extracted_data = {}
            for field_name in fields:
                if field_name in doc:
                    value = doc[field_name]
                    # Handle field transformations for display
                    if field_keyword(field_name) and isinstance(value, str) and "###" in value:
                        extracted_data[field_name] = value.split("###")
                    elif field_name == "position_int" and isinstance(value, str) and "_" in value:
                        try:
                            arr = [int(hex_val, 16) for hex_val in value.split('_')]
                            extracted_data[field_name] = [arr[i:i + 5] for i in range(0, len(arr), 5)]
                        except ValueError:
                            extracted_data[field_name] = value
                    elif field_name in ["page_num_int", "top_int"] and isinstance(value, str) and "_" in value:
                        try:
                            extracted_data[field_name] = [int(hex_val, 16) for hex_val in value.split('_')]
                        except ValueError:
                            extracted_data[field_name] = value
                    else:
                        extracted_data[field_name] = value

            if extracted_data:
                res_fields[doc_id] = extracted_data

        return res_fields

    def getHighlight(self, res, keywords: list[str], fieldnm: str) -> dict[str, str]:
        """Milvus does not provide built-in highlighting"""
        return {}

    def getAggregation(self, res, fieldnm: str) -> list[tuple[str, int]]:
        """Milvus does not offer direct aggregation capabilities"""
        return []

    """
    SQL
    """

    def sql(self, sql: str, fetch_size: int, format: str):
        """Milvus does not support SQL queries directly"""
        logger.warning("Milvus does not support SQL queries directly. Use search/query methods.")
        raise NotImplementedError("Milvus does not support SQL queries directly. Use search/query methods.")
