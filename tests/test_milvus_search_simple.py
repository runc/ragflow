#!/usr/bin/env python3
"""
简化的 Milvus 搜索测试脚本
"""

import os
import sys
import logging

# 设置环境变量
os.environ['DOC_ENGINE'] = 'milvus'

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ragflow'))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_milvus_basic():
    """测试 Milvus 基本功能"""
    try:
        # 导入必要的模块
        from pymilvus import connections, utility, Collection
        
        logger.info("=== 测试 Milvus 基本连接 ===")
        
        # 直接连接 Milvus
        connections.connect(
            alias="test_connection",
            host="localhost",
            port="19530"
        )
        
        # 列出所有集合
        collections = utility.list_collections(using="test_connection")
        logger.info(f"找到 {len(collections)} 个集合:")
        for collection_name in collections:
            logger.info(f"  - {collection_name}")
        
        # 检查每个集合的详细信息
        for collection_name in collections[:3]:  # 只检查前3个
            try:
                collection = Collection(name=collection_name, using="test_connection")
                collection.load()
                
                logger.info(f"\n集合 {collection_name}:")
                logger.info(f"  实体数量: {collection.num_entities}")
                
                # 显示字段信息
                logger.info("  字段信息:")
                vector_fields = []
                scalar_fields = []
                
                for field in collection.schema.fields:
                    field_info = f"    {field.name}: {field.dtype}"
                    if hasattr(field, 'params') and field.params:
                        field_info += f" {field.params}"
                    logger.info(field_info)
                    
                    if 'VECTOR' in str(field.dtype):
                        vector_fields.append(field.name)
                    else:
                        scalar_fields.append(field.name)
                
                # 如果有数据，尝试查询一些样本
                if collection.num_entities > 0:
                    logger.info("  样本数据:")
                    try:
                        # 查询前2条记录
                        results = collection.query(
                            expr="",
                            output_fields=scalar_fields[:5],  # 只取前5个标量字段
                            limit=2
                        )
                        
                        for i, result in enumerate(results):
                            logger.info(f"    记录 {i+1}: {result}")
                            
                    except Exception as e:
                        logger.warning(f"    查询样本数据失败: {e}")
                
                # 如果有向量字段，尝试向量搜索
                if vector_fields and collection.num_entities > 0:
                    logger.info(f"  测试向量搜索 (向量字段: {vector_fields[0]}):")
                    try:
                        # 获取向量维度
                        vector_field = None
                        for field in collection.schema.fields:
                            if field.name == vector_fields[0]:
                                vector_field = field
                                break
                        
                        if vector_field and hasattr(vector_field, 'params') and 'dim' in vector_field.params:
                            dim = vector_field.params['dim']
                            logger.info(f"    向量维度: {dim}")
                            
                            # 创建测试向量
                            test_vector = [0.1 * i for i in range(dim)]
                            
                            # 执行搜索
                            search_results = collection.search(
                                data=[test_vector],
                                anns_field=vector_fields[0],
                                param={"metric_type": "L2", "params": {"nprobe": 10}},
                                limit=3,
                                output_fields=scalar_fields[:3]  # 只返回前3个标量字段
                            )
                            
                            if search_results and len(search_results) > 0:
                                hits = search_results[0]
                                logger.info(f"    搜索结果: {len(hits)} 条")
                                for j, hit in enumerate(hits[:2]):  # 只显示前2条
                                    logger.info(f"      结果 {j+1}: 距离={hit.distance}, 数据={hit.entity.to_dict()}")
                            else:
                                logger.warning("    搜索无结果")
                                
                    except Exception as e:
                        logger.warning(f"    向量搜索失败: {e}")
                        
            except Exception as e:
                logger.error(f"检查集合 {collection_name} 失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Milvus 基本测试失败: {e}")
        return False

def test_ragflow_milvus():
    """测试 RAGFlow 的 Milvus 连接"""
    try:
        logger.info("\n=== 测试 RAGFlow Milvus 连接 ===")
        
        # 导入 RAGFlow 的 Milvus 连接
        from rag.utils.milvus_conn import MilvusConnection
        
        # 创建连接
        conn = MilvusConnection()
        logger.info("✓ RAGFlow Milvus 连接创建成功")
        
        # 检查健康状态
        health = conn.health()
        logger.info(f"✓ 健康状态: {health}")
        
        return conn
        
    except Exception as e:
        logger.error(f"RAGFlow Milvus 连接测试失败: {e}")
        return None

def test_search_with_ragflow(conn):
    """使用 RAGFlow 连接测试搜索"""
    try:
        logger.info("\n=== 测试 RAGFlow 搜索功能 ===")
        
        from rag.utils.doc_store_conn import MatchDenseExpr, OrderByExpr
        from pymilvus import utility
        
        # 获取所有集合
        collections = utility.list_collections(using=conn.alias)
        if not collections:
            logger.warning("没有找到任何集合")
            return
        
        # 使用第一个集合进行测试
        test_collection_name = collections[0]
        logger.info(f"使用集合进行测试: {test_collection_name}")
        
        # 解析集合名称
        parts = test_collection_name.split('_')
        if len(parts) < 2:
            logger.error(f"集合名称格式不正确: {test_collection_name}")
            return
        
        index_name = '_'.join(parts[:-1])
        kb_id = parts[-1]
        logger.info(f"解析结果 - 索引名: {index_name}, 知识库ID: {kb_id}")
        
        # 测试不同维度的向量搜索
        for vector_size in [512, 768, 1024, 1536]:
            try:
                logger.info(f"\n测试 {vector_size} 维向量搜索...")
                
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
                    selectFields=["id", "doc_id", "kb_id"],
                    highlightFields=[],
                    condition={},
                    matchExprs=[match_expr],
                    orderBy=OrderByExpr(),
                    offset=0,
                    limit=5,
                    indexNames=[index_name],
                    knowledgebaseIds=[kb_id]
                )
                
                total = conn.getTotal(results)
                chunk_ids = conn.getChunkIds(results)
                
                logger.info(f"  搜索结果:")
                logger.info(f"    总数: {total}")
                logger.info(f"    返回条目: {len(chunk_ids)}")
                
                if total > 0:
                    logger.info(f"    示例ID: {chunk_ids[:3]}")
                    logger.info(f"  ✓ {vector_size}维搜索成功!")
                    break
                else:
                    logger.warning(f"  {vector_size}维搜索无结果")
                    
            except Exception as e:
                logger.warning(f"  {vector_size}维搜索失败: {e}")
                
    except Exception as e:
        logger.error(f"RAGFlow 搜索测试失败: {e}")

def main():
    """主函数"""
    logger.info("开始 Milvus 搜索问题诊断...")
    
    # 1. 测试基本 Milvus 连接
    if not test_milvus_basic():
        logger.error("基本 Milvus 测试失败，退出")
        return
    
    # 2. 测试 RAGFlow Milvus 连接
    conn = test_ragflow_milvus()
    if not conn:
        logger.error("RAGFlow Milvus 连接失败，退出")
        return
    
    # 3. 测试搜索功能
    test_search_with_ragflow(conn)
    
    logger.info("\n诊断完成")

if __name__ == "__main__":
    main()
