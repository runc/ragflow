import pytest
import random
import string
import copy
import pymilvus
from pymilvus import DataType
from rag.settings import settings
from rag.utils.doc_store_conn import DocStoreConnection

# 生成随机tenant和kb_id，避免污染线上数据
def random_id(prefix):
    return f"{prefix}_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

# 直接创建MilvusConnection类，绕过singleton装饰器
class TestMilvusConnection(DocStoreConnection):
    def __init__(self, tenant: str):
        super().__init__(tenant)
        self.conn = self._create_connection()
        # Sanitize tenant ID for db name and apply length considerations
        db_tenant_suffix = tenant.replace("-", "_")
        self.db_name = f"rag_{db_tenant_suffix}"
        self._create_db_if_not_exists()

    def _create_connection(self):
        try:
            milvus_config = settings.MILVUS
            host = milvus_config.get("host", "localhost")
            port = milvus_config.get("port", "19530")
        except AttributeError:
            host = "localhost"
            port = "19530"

        try:
            conn_params = {
                "alias": f"default_{self.tenant}",
                "host": host,
                "port": port,
            }
            conn = pymilvus.connections.connect(**conn_params)
            return conn
        except Exception as e:
            raise

    def _create_db_if_not_exists(self):
        try:
            if not pymilvus.utility.has_database(self.db_name, using=f"default_{self.tenant}"):
                pymilvus.utility.create_database(self.db_name, using=f"default_{self.tenant}")
        except Exception as e:
            raise

    @property
    def dbType(self) -> str:
        return "milvus"

    def health(self) -> bool:
        try:
            pymilvus.connections.get_connection(alias=f"default_{self.tenant}")
            pymilvus.utility.list_collections(using=f"default_{self.tenant}", db_name=self.db_name)
            return True
        except Exception as e:
            return False

    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):
        from pymilvus import CollectionSchema, FieldSchema, Collection

        collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")
        vector_field_name = f"q_{vectorSize}_vec"

        if self.indexExist(indexName, knowledgebaseId):
            return

        try:
            MAX_VARCHAR_LENGTH = 1024
            ID_VARCHAR_LENGTH = 255

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=ID_VARCHAR_LENGTH, description="Primary key chunk ID"),
                FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=ID_VARCHAR_LENGTH, description="Knowledge base ID"),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=ID_VARCHAR_LENGTH, description="Document ID"),
                FieldSchema(name="docnm_kwd", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH, description="Document name keyword"),
                FieldSchema(name="content_ltks", dtype=DataType.VARCHAR, max_length=65535, description="Content long tokens"),
                FieldSchema(name="name_tks", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH, description="Name tokens"),
                FieldSchema(name="important_kwd", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH, description="Important keywords"),
                FieldSchema(name="question_tks", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LENGTH, description="Question tokens"),
                FieldSchema(name="page_num_int", dtype=DataType.INT32, description="Page number"),
                FieldSchema(name="create_timestamp_flt", dtype=DataType.FLOAT, description="Creation timestamp"),
                FieldSchema(name=vector_field_name, dtype=DataType.FLOAT_VECTOR, dim=vectorSize, description="Query vector embedding")
            ]

            schema = CollectionSchema(fields=fields, description=f"Collection for {indexName} of KB {knowledgebaseId}", enable_dynamic_field=True)

            collection = Collection(
                name=collection_name,
                schema=schema,
                using=f"default_{self.tenant}",
                db_name=self.db_name
            )

            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            collection.create_index(field_name=vector_field_name, index_params=index_params)
            collection.load()

        except Exception as e:
            raise

    def indexExist(self, indexName: str, knowledgebaseId: str = None) -> bool:
        if knowledgebaseId:
            collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")
        else:
            collection_name = indexName

        try:
            return pymilvus.utility.has_collection(
                collection_name=collection_name,
                using=f"default_{self.tenant}",
                db_name=self.db_name
            )
        except Exception as e:
            return False

    def insert(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        from pymilvus import Collection
        collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")

        if not self.indexExist(indexName, knowledgebaseId):
            raise ValueError(f"Collection {collection_name} does not exist.")

        try:
            collection = Collection(name=collection_name, using=f"default_{self.tenant}", db_name=self.db_name)
            collection.load()

            vector_field_name = None
            for field in collection.schema.fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    vector_field_name = field.name
                    break
            if not vector_field_name:
                raise ValueError(f"No float vector field found in schema for collection {collection_name}")

            prepared_docs = []
            for doc in documents:
                prepared_doc = copy.deepcopy(doc)

                if 'embedding' in prepared_doc and vector_field_name not in prepared_doc:
                    prepared_doc[vector_field_name] = prepared_doc.pop('embedding')
                elif vector_field_name not in prepared_doc:
                     raise ValueError(f"Vector data not found in document under key 'embedding' or '{vector_field_name}' for doc ID {doc.get('id')}")

                if 'kb_id' not in prepared_doc and knowledgebaseId:
                     prepared_doc['kb_id'] = knowledgebaseId

                prepared_docs.append(prepared_doc)

            if not prepared_docs:
                return []

            mutation_result = collection.insert(prepared_docs)
            collection.flush()
            return []

        except Exception as e:
            raise

    def get(self, chunkId: str, indexName: str, knowledgebaseIds: list[str]) -> dict | None:
        from pymilvus import Collection
        if not isinstance(knowledgebaseIds, list):
            knowledgebaseIds = [knowledgebaseIds]

        for kb_id in knowledgebaseIds:
            collection_name = f"{indexName}_{kb_id}".replace("-", "_")
            if not self.indexExist(indexName, kb_id):
                continue

            try:
                collection = Collection(name=collection_name, using=f"default_{self.tenant}", db_name=self.db_name)
                collection.load()

                expr = f"id == '{chunkId}'"
                schema_fields = [field.name for field in collection.schema.fields]

                results = collection.query(
                    expr=expr,
                    output_fields=schema_fields,
                    limit=1
                )

                if results:
                    return results[0]
            except Exception as e:
                continue

        return None

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        from pymilvus import Collection
        collection_name = f"{indexName}_{knowledgebaseId}".replace("-", "_")

        if not self.indexExist(indexName, knowledgebaseId):
            return 0

        try:
            collection = Collection(name=collection_name, using=f"default_{self.tenant}", db_name=self.db_name)
            collection.load()

            expr_parts = []
            if "id" in condition:
                ids = condition["id"]
                if isinstance(ids, list):
                    if not ids: return 0
                    formatted_ids = [f"'{str(i)}'" if isinstance(i, str) else str(i) for i in ids]
                    expr_parts.append(f"id in [{', '.join(formatted_ids)}]")
                elif isinstance(ids, str):
                    expr_parts.append(f"id == '{ids}'")

            if not expr_parts:
                return 0

            expr = " and ".join(expr_parts)
            mutation_result = collection.delete(expr)
            collection.flush()
            return mutation_result.delete_count

        except Exception as e:
            raise

    def search(self, selectFields: list[str], highlightFields: list[str], condition: dict,
               matchExprs: list, orderBy: object, offset: int, limit: int,
               indexNames: str | list[str], knowledgebaseIds: list[str],
               aggFields: list[str] = [], rank_feature: dict | None = None) -> list[dict]:
        from pymilvus import Collection, DataType
        from rag.utils.doc_store_conn import MatchDenseExpr

        if not knowledgebaseIds:
            return []

        current_index_name = indexNames[0] if isinstance(indexNames, list) else indexNames
        results_list = []

        for kb_id in knowledgebaseIds:
            collection_name = f"{current_index_name}_{kb_id}".replace("-", "_")
            if not self.indexExist(current_index_name, kb_id):
                continue

            try:
                collection = Collection(name=collection_name, using=f"default_{self.tenant}", db_name=self.db_name)
                collection.load()

                for expr in matchExprs:
                    if isinstance(expr, MatchDenseExpr):
                        vector_to_search = expr.data
                        vec_field_name = None
                        for field_schema in collection.schema.fields:
                            if field_schema.dtype == DataType.FLOAT_VECTOR:
                                vec_field_name = field_schema.name
                                break
                        if not vec_field_name:
                            continue

                        search_params = {
                            "metric_type": "L2",
                            "params": {"nprobe": 10},
                        }

                        actual_output_fields = []
                        if not selectFields or "*" in selectFields:
                            actual_output_fields = [f.name for f in collection.schema.fields if f.dtype != DataType.FLOAT_VECTOR]
                            if "id" not in actual_output_fields:
                                actual_output_fields.append("id")
                        else:
                            actual_output_fields = list(set(selectFields + ["id"]))

                        search_results = collection.search(
                            data=[vector_to_search],
                            anns_field=vec_field_name,
                            param=search_params,
                            limit=limit + offset,
                            expr="",
                            output_fields=actual_output_fields,
                            consistency_level="Strong"
                        )

                        hits = search_results[0]
                        paginated_hits = hits[offset : offset + limit]

                        for hit in paginated_hits:
                            result_doc = hit.entity.to_dict()
                            result_doc['score'] = hit.distance
                            results_list.append(result_doc)

            except Exception as e:
                continue

        return results_list

def test_milvus_connection_health():
    tenant = random_id("testtenant")
    conn = TestMilvusConnection(tenant)
    assert conn.health() is True

def test_create_and_check_index():
    tenant = random_id("testtenant")
    conn = TestMilvusConnection(tenant)
    index_name = "testidx"
    kb_id = random_id("kb")
    vector_size = 8
    conn.createIdx(index_name, kb_id, vector_size)
    assert conn.indexExist(index_name, kb_id)

def test_insert_and_get():
    tenant = random_id("testtenant")
    conn = TestMilvusConnection(tenant)
    index_name = "testidx"
    kb_id = random_id("kb")
    vector_size = 8
    conn.createIdx(index_name, kb_id, vector_size)
    doc_id = random_id("doc")
    doc = {
        "id": doc_id,
        "kb_id": kb_id,
        "doc_id": "doc1",
        "docnm_kwd": "测试文档",
        "content_ltks": "内容token",
        "name_tks": "名称token",
        "important_kwd": "关键词",
        "question_tks": "问题token",
        "page_num_int": 1,
        "create_timestamp_flt": 123456.0,
        "embedding": [float(i) for i in range(vector_size)]
    }
    conn.insert([doc], index_name, kb_id)
    res = conn.get(doc_id, index_name, [kb_id])
    assert res is not None
    assert res["id"] == doc_id

def test_delete():
    tenant = random_id("testtenant")
    conn = TestMilvusConnection(tenant)
    index_name = "testidx"
    kb_id = random_id("kb")
    vector_size = 8
    conn.createIdx(index_name, kb_id, vector_size)
    doc_id = random_id("doc")
    doc = {
        "id": doc_id,
        "kb_id": kb_id,
        "doc_id": "doc1",
        "docnm_kwd": "测试文档",
        "content_ltks": "内容token",
        "name_tks": "名称token",
        "important_kwd": "关键词",
        "question_tks": "问题token",
        "page_num_int": 1,
        "create_timestamp_flt": 123456.0,
        "embedding": [float(i) for i in range(vector_size)]
    }
    conn.insert([doc], index_name, kb_id)
    deleted = conn.delete({"id": doc_id}, index_name, kb_id)
    assert deleted >= 1
    res = conn.get(doc_id, index_name, [kb_id])
    assert res is None

def test_search():
    tenant = random_id("testtenant")
    conn = TestMilvusConnection(tenant)
    index_name = "testidx"
    kb_id = random_id("kb")
    vector_size = 8
    conn.createIdx(index_name, kb_id, vector_size)
    # 插入两条数据
    docs = []
    for i in range(2):
        doc_id = random_id(f"doc{i}")
        docs.append({
            "id": doc_id,
            "kb_id": kb_id,
            "doc_id": f"doc{i}",
            "docnm_kwd": "测试文档",
            "content_ltks": "内容token",
            "name_tks": "名称token",
            "important_kwd": "关键词",
            "question_tks": "问题token",
            "page_num_int": i,
            "create_timestamp_flt": 123456.0 + i,
            "embedding": [float(i) for i in range(vector_size)]
        })
    conn.insert(docs, index_name, kb_id)
    # 构造向量检索表达式
    from rag.utils.doc_store_conn import MatchDenseExpr
    query_vec = [0.0 for _ in range(vector_size)]
    match_expr = MatchDenseExpr(field=None, data=query_vec)
    results = conn.search(
        selectFields=["id", "doc_id"],
        highlightFields=[],
        condition={},
        matchExprs=[match_expr],
        orderBy=None,
        offset=0,
        limit=2,
        indexNames=index_name,
        knowledgebaseIds=[kb_id]
    )
    assert isinstance(results, list)
    assert len(results) >= 1
    assert "id" in results[0]
