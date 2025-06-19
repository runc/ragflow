#!/usr/bin/env python3
"""
测试 RAGFlow 的 MilvusConnection.search 方法
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from api import settings
from rag.utils.milvus_conn import MilvusConnection
from rag.nlp.search import MatchDenseExpr
import numpy as np

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_milvus_connection_search():
    """直接测试 MilvusConnection 的 search 方法"""
    
    # 初始化设置
    settings.init_settings()
    
    # 目标参数
    doc_id = "a106ea544ce911f0ae6b5df10d5df26e"
    kb_id = "a4eca9a4cecd11ef8a045df10d5df26e"
    
    try:
        # 创建 Milvus 连接（RAGFlow 方式）
        milvus_conn = MilvusConnection()
        
        print(f"=== 测试 MilvusConnection.search ===")
        print(f"目标 doc_id: {doc_id}")
        print(f"目标 kb_id: {kb_id}")
        
        # 创建测试向量（768维零向量，仅用于测试过滤条件）
        test_embedding = np.zeros(768, dtype=np.float32).tolist()
        
        # 构造查询条件
        condition = {"doc_id": doc_id}
        
        # 构造 MatchDenseExpr（使用正确的参数）
        match_expr = MatchDenseExpr(
            vector_column_name="q_768_vec",  # 根据向量维度命名
            embedding_data=test_embedding,
            embedding_data_type='float',
            distance_type='cosine',
            topn=100,
            extra_options={"similarity": 0.1}
        )
        
        # 调用 search 方法
        print(f"\n=== 调用 MilvusConnection.search ===")
        print(f"condition: {condition}")
        print(f"offset: 0, limit: 10")
        
        results = milvus_conn.search(
            selectFields=["*"],
            highlightFields=[],
            condition=condition,
            matchExprs=[match_expr],
            orderBy=None,
            offset=0,
            limit=10,
            indexNames="knowledge_graph",
            knowledgebaseIds=[kb_id],
            aggFields=[],
            rank_feature=None
        )
        
        print(f"\n=== 搜索结果 ===")
        print(f"返回结果: {results}")
        print(f"hits 数量: {len(results.get('hits', {}).get('hits', []))}")
        print(f"total value: {results.get('hits', {}).get('total', {}).get('value', 0)}")
        
        # 如果有结果，打印前几个
        hits = results.get('hits', {}).get('hits', [])
        for i, hit in enumerate(hits[:3]):
            print(f"Hit {i+1}: id={hit.get('_id')}, score={hit.get('_score')}")
            source = hit.get('_source', {})
            print(f"  doc_id: {source.get('doc_id')}")
            print(f"  content: {source.get('content_with_weight', '')[:100]}...")
        
        # 测试不同的查询条件
        print(f"\n=== 测试其他查询条件 ===")
        
        # 测试1：不带过滤条件
        print("1. 不带过滤条件的查询:")
        results_no_filter = milvus_conn.search(
            selectFields=["*"],
            highlightFields=[],
            condition={},
            matchExprs=[match_expr],
            orderBy=None,
            offset=0,
            limit=5,
            indexNames="knowledge_graph",
            knowledgebaseIds=[kb_id],
            aggFields=[],
            rank_feature=None
        )
        print(f"   结果数量: {len(results_no_filter.get('hits', {}).get('hits', []))}")
        print(f"   total: {results_no_filter.get('hits', {}).get('total', {}).get('value', 0)}")
        
        # 测试2：更大的 limit
        print("2. 更大 limit 的查询:")
        results_large_limit = milvus_conn.search(
            selectFields=["*"],
            highlightFields=[],
            condition=condition,
            matchExprs=[match_expr],
            orderBy=None,
            offset=0,
            limit=100,
            indexNames="knowledge_graph",
            knowledgebaseIds=[kb_id],
            aggFields=[],
            rank_feature=None
        )
        print(f"   结果数量: {len(results_large_limit.get('hits', {}).get('hits', []))}")
        print(f"   total: {results_large_limit.get('hits', {}).get('total', {}).get('value', 0)}")
        
        return results
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_milvus_connection_search()
