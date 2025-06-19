#!/usr/bin/env python3
"""
简单的 Milvus doc_id 验证脚本
"""

import os
import sys

# 设置环境变量
os.environ['DOC_ENGINE'] = 'milvus'

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ragflow'))

print("开始验证...")

try:
    from rag.utils.milvus_conn import MilvusConnection
    from pymilvus import Collection
    
    # 创建连接
    conn = MilvusConnection()
    print("连接成功")
    
    # 使用已知有数据的集合
    collection_name = "ragflow_7179adc24c1b11f0bb2a6b89a3fc27c2_7b5bf8b84c1b11f0bb2a6b89a3fc27c2"
    target_doc_id = "a106ea544ce911f0ae6b5df10d5df26e"
    
    print(f"测试集合: {collection_name}")
    print(f"目标文档ID: {target_doc_id}")
    
    # 获取集合信息
    collection = Collection(name=collection_name, using=conn.alias)
    collection.load()
    
    print(f"集合实体数: {collection.num_entities}")
    
    # 查询前5个文档的 doc_id
    from pymilvus import connections
    results = collection.query(
        expr="",  # 无过滤条件
        output_fields=["id", "doc_id", "kb_id"],
        limit=5
    )
    
    print(f"查询结果数量: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"结果 {i+1}:")
        print(f"  id: {result['id']}")
        print(f"  doc_id: '{result['doc_id']}'")
        print(f"  kb_id: '{result['kb_id']}'")
        print(f"  doc_id == target: {result['doc_id'] == target_doc_id}")
    
    # 尝试过滤查询
    print(f"\n--- 尝试过滤查询 ---")
    filtered_results = collection.query(
        expr=f"doc_id == '{target_doc_id}'",
        output_fields=["id", "doc_id", "kb_id"],
        limit=10
    )
    
    print(f"过滤查询结果数量: {len(filtered_results)}")
    for result in filtered_results:
        print(f"  id: {result['id']}, doc_id: '{result['doc_id']}'")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print("验证完成")
