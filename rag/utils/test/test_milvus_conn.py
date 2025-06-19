import pytest
import random
import string
import traceback
from rag.utils.milvus_conn import MilvusConnection

def random_id(prefix):
    return f"{prefix}_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

def test_milvus_connection_health():
    tenant = random_id("testtenant")
    try:
        conn = MilvusConnection(tenant)
        health_result = conn.health()
        print(f"[health] tenant={tenant} result={health_result}")
        assert health_result is True
    except Exception as e:
        print(f"[health] Exception: {e}\n{traceback.format_exc()}")
        assert False

def test_create_and_check_index():
    tenant = random_id("testtenant")
    index_name = "testidx"
    kb_id = random_id("kb")
    vector_size = 8
    try:
        conn = MilvusConnection(tenant)
        conn.createIdx(index_name, kb_id, vector_size)
        exist = conn.indexExist(index_name, kb_id)
        print(f"[createIdx] tenant={tenant} index={index_name} kb_id={kb_id} exist={exist}")
        assert exist
    except Exception as e:
        print(f"[createIdx] Exception: {e}\n{traceback.format_exc()}")
        assert False

def test_insert_and_get():
    tenant = random_id("testtenant")
    index_name = "testidx"
    kb_id = random_id("kb")
    vector_size = 8
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
    try:
        conn = MilvusConnection(tenant)
        conn.createIdx(index_name, kb_id, vector_size)
        conn.insert([doc], index_name, kb_id)
        res = conn.get(doc_id, index_name, [kb_id])
        print(f"[insert/get] tenant={tenant} doc_id={doc_id} result={res}")
        assert res is not None
        assert res["id"] == doc_id
    except Exception as e:
        print(f"[insert/get] Exception: {e}\n{traceback.format_exc()}")
        assert False

def test_delete():
    tenant = random_id("testtenant")
    index_name = "testidx"
    kb_id = random_id("kb")
    vector_size = 8
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
    try:
        conn = MilvusConnection(tenant)
        conn.createIdx(index_name, kb_id, vector_size)
        conn.insert([doc], index_name, kb_id)
        deleted = conn.delete({"id": doc_id}, index_name, kb_id)
        print(f"[delete] tenant={tenant} doc_id={doc_id} deleted={deleted}")
        assert deleted >= 1
        res = conn.get(doc_id, index_name, [kb_id])
        assert res is None
    except Exception as e:
        print(f"[delete] Exception: {e}\n{traceback.format_exc()}")
        assert False

def test_search():
    tenant = random_id("testtenant")
    index_name = "testidx"
    kb_id = random_id("kb")
    vector_size = 8
    try:
        conn = MilvusConnection(tenant)
        conn.createIdx(index_name, kb_id, vector_size)
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
        from rag.utils.doc_store_conn import MatchDenseExpr
        query_vec = [0.0 for _ in range(vector_size)]
        match_expr = MatchDenseExpr(
            vector_column_name=f"q_{vector_size}_vec",
            embedding_data=query_vec,
            embedding_data_type="float",
            distance_type="L2",
            topn=2
        )
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
        print(f"[search] tenant={tenant} results={results}")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert "id" in results[0]
    except Exception as e:
        print(f"[search] Exception: {e}\n{traceback.format_exc()}")
        assert False
