import pytest
import random
import string
from rag.utils.milvus_conn import MilvusConnection

# 生成随机tenant和kb_id，避免污染线上数据
def random_id(prefix):
    return f"{prefix}_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

def test_milvus_connection_health():
    tenant = random_id("testtenant")
    conn = MilvusConnection(tenant)
    assert conn.health() is True

def test_create_and_check_index():
    tenant = random_id("testtenant")
    conn = MilvusConnection(tenant)
    index_name = "testidx"
    kb_id = random_id("kb")
    vector_size = 8
    conn.createIdx(index_name, kb_id, vector_size)
    assert conn.indexExist(index_name, kb_id)

def test_insert_and_get():
    tenant = random_id("testtenant")
    conn = MilvusConnection(tenant)
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
    conn = MilvusConnection(tenant)
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
    conn = MilvusConnection(tenant)
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
