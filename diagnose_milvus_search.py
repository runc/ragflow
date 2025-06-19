#!/usr/bin/env python3
"""
Milvus 搜索问题诊断脚本
用于检查 Milvus 连接、数据存在性和搜索功能
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ragflow'))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_milvus_connection():
    """检查 Milvus 连接"""
    try:
        from rag.utils.milvus_conn import MilvusConnection
        from pymilvus import utility
        
        logger.info("=== 检查 Milvus 连接 ===")
        
        # 创建连接
        conn = MilvusConnection()
        logger.info(f"✓ Milvus 连接创建成功")
        
        # 检查健康状态
        health = conn.health()
        logger.info(f"✓ Milvus 健康状态: {health}")
        
        # 列出所有集合
        collections = utility.list_collections(using=conn.alias)
        logger.info(f"✓ 可用集合数量: {len(collections)}")
        for collection_name in collections:
            logger.info(f"  - {collection_name}")
            
        return conn, collections
        
    except Exception as e:
        logger.error(f"✗ Milvus 连接失败: {e}")
        return None, []

def check_collection_details(conn, collection_names: List[str]):
    """检查集合详细信息"""
    try:
        from pymilvus import Collection
        
        logger.info("=== 检查集合详细信息 ===")
        
        for collection_name in collection_names:
            try:
                collection = Collection(name=collection_name, using=conn.alias)
                collection.load()
                
                logger.info(f"集合: {collection_name}")
                logger.info(f"  - 实体数量: {collection.num_entities}")
                logger.info(f"  - 字段列表:")
                
                for field in collection.schema.fields:
                    logger.info(f"    * {field.name}: {field.dtype} {getattr(field, 'params', {})}")
                
                # 检查索引
                indexes = collection.indexes
                logger.info(f"  - 索引数量: {len(indexes)}")
                for index in indexes:
                    logger.info(f"    * 字段: {index.field_name}, 类型: {index.params}")
                
                # 检查加载状态
                load_state = utility.load_state(collection_name, using=conn.alias)
                logger.info(f"  - 加载状态: {load_state}")
                
            except Exception as e:
                logger.error(f"  ✗ 检查集合 {collection_name} 失败: {e}")
                
    except Exception as e:
        logger.error(f"✗ 检查集合详细信息失败: {e}")

def test_simple_search(conn, collection_names: List[str]):
    """测试简单搜索"""
    try:
        from rag.utils.doc_store_conn import MatchDenseExpr, OrderByExpr
        
        logger.info("=== 测试简单搜索 ===")
        
        if not collection_names:
            logger.warning("没有可用的集合进行搜索测试")
            return
            
        # 使用第一个集合进行测试
        test_collection_name = collection_names[0]
        logger.info(f"使用集合进行测试: {test_collection_name}")
        
        # 解析集合名称获取索引名和知识库ID
        # 格式: indexName_knowledgebaseId
        parts = test_collection_name.split('_')
        if len(parts) < 2:
            logger.error(f"集合名称格式不正确: {test_collection_name}")
            return
            
        index_name = '_'.join(parts[:-1])
        kb_id = parts[-1]
        
        logger.info(f"解析结果 - 索引名: {index_name}, 知识库ID: {kb_id}")
        
        # 创建测试向量（假设是1024维）
        vector_sizes = [512, 768, 1024, 1536]
        
        for vector_size in vector_sizes:
            try:
                logger.info(f"测试 {vector_size} 维向量搜索...")
                
                # 创建测试向量
                test_vector = [0.1 * i for i in range(vector_size)]
                
                # 创建匹配表达式
                match_expr = MatchDenseExpr(
                    vector_column_name=f"q_{vector_size}_vec",
                    embedding_data=test_vector,
                    embedding_data_type="float",
                    distance_type="L2",
                    topn=5
                )
                
                # 执行搜索
                results = conn.search(
                    selectFields=["id", "doc_id", "kb_id", "content_ltks"],
                    highlightFields=[],
                    condition={},
                    matchExprs=[match_expr],
                    orderBy=OrderByExpr(),
                    offset=0,
                    limit=5,
                    indexNames=[index_name],
                    knowledgebaseIds=[kb_id]
                )
                
                logger.info(f"  ✓ {vector_size}维搜索结果:")
                logger.info(f"    - 总数: {conn.getTotal(results)}")
                logger.info(f"    - 返回条目: {len(conn.getChunkIds(results))}")
                
                if conn.getTotal(results) > 0:
                    chunk_ids = conn.getChunkIds(results)[:3]  # 只显示前3个
                    logger.info(f"    - 示例ID: {chunk_ids}")
                    break  # 找到有结果的向量维度就停止
                    
            except Exception as e:
                logger.warning(f"  ✗ {vector_size}维搜索失败: {e}")
                continue
                
    except Exception as e:
        logger.error(f"✗ 测试搜索失败: {e}")

def check_data_samples(conn, collection_names: List[str]):
    """检查数据样本"""
    try:
        from pymilvus import Collection
        
        logger.info("=== 检查数据样本 ===")
        
        for collection_name in collection_names[:2]:  # 只检查前2个集合
            try:
                collection = Collection(name=collection_name, using=conn.alias)
                collection.load()
                
                if collection.num_entities == 0:
                    logger.warning(f"集合 {collection_name} 为空")
                    continue
                
                logger.info(f"集合 {collection_name} 数据样本:")
                
                # 获取所有非向量字段
                output_fields = []
                vector_fields = []
                for field in collection.schema.fields:
                    if 'VECTOR' in str(field.dtype):
                        vector_fields.append(field.name)
                    else:
                        output_fields.append(field.name)
                
                # 查询前3条记录
                results = collection.query(
                    expr="",  # 空表达式获取所有数据
                    output_fields=output_fields,
                    limit=3
                )
                
                logger.info(f"  - 非向量字段: {output_fields}")
                logger.info(f"  - 向量字段: {vector_fields}")
                logger.info(f"  - 样本数据 ({len(results)} 条):")
                
                for i, result in enumerate(results):
                    logger.info(f"    记录 {i+1}:")
                    for key, value in result.items():
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        logger.info(f"      {key}: {value}")
                        
            except Exception as e:
                logger.error(f"  ✗ 检查集合 {collection_name} 数据样本失败: {e}")
                
    except Exception as e:
        logger.error(f"✗ 检查数据样本失败: {e}")

def main():
    """主函数"""
    logger.info("开始 Milvus 搜索问题诊断...")
    
    # 1. 检查连接
    conn, collections = check_milvus_connection()
    if not conn:
        logger.error("无法建立 Milvus 连接，退出诊断")
        return
    
    # 2. 检查集合详细信息
    if collections:
        check_collection_details(conn, collections)
        
        # 3. 检查数据样本
        check_data_samples(conn, collections)
        
        # 4. 测试搜索
        test_simple_search(conn, collections)
    else:
        logger.warning("没有找到任何集合")
    
    logger.info("诊断完成")

if __name__ == "__main__":
    main()
