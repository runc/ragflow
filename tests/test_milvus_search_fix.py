#!/usr/bin/env python3
"""
测试 Milvus 搜索修复的脚本
"""

import os
import sys
import logging
import json

# 设置环境变量
os.environ['DOC_ENGINE'] = 'milvus'

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ragflow'))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_milvus_search_fix():
    """测试 Milvus 搜索修复"""
    try:
        logger.info("=== 测试 Milvus 搜索修复 ===")
        
        # 导入必要的模块
        from rag.utils.milvus_conn import MilvusConnection
        from rag.utils.doc_store_conn import MatchDenseExpr, OrderByExpr
        from pymilvus import utility
        
        # 创建连接
        conn = MilvusConnection()
        logger.info("✓ Milvus 连接创建成功")
        
        # 获取所有集合
        collections = utility.list_collections(using=conn.alias)
        logger.info(f"✓ 找到 {len(collections)} 个集合")
        
        if not collections:
            logger.warning("没有找到任何集合，无法进行搜索测试")
            return False
        
        # 找到有数据的集合
        test_collection_name = None
        for collection_name in collections:
            from pymilvus import Collection
            collection = Collection(name=collection_name, using=conn.alias)
            collection.load()
            if collection.num_entities > 0:
                test_collection_name = collection_name
                break
        
        if not test_collection_name:
            logger.warning("没有找到有数据的集合，无法进行搜索测试")
            return False
        
        logger.info(f"使用有数据的集合进行测试: {test_collection_name}")
        
        # 解析集合名称
        parts = test_collection_name.split('_')
        if len(parts) < 2:
            logger.error(f"集合名称格式不正确: {test_collection_name}")
            return False
        
        index_name = '_'.join(parts[:-1])
        kb_id = parts[-1]
        logger.info(f"解析结果 - 索引名: {index_name}, 知识库ID: {kb_id}")
        
        # 检查集合是否有数据
        from pymilvus import Collection
        collection = Collection(name=test_collection_name, using=conn.alias)
        collection.load()
        
        entity_count = collection.num_entities
        logger.info(f"集合 {test_collection_name} 包含 {entity_count} 个实体")
        
        if entity_count == 0:
            logger.warning("集合为空，无法进行搜索测试")
            return False
        
        # 获取集合的向量字段信息
        from pymilvus import DataType
        vector_fields = []
        for field in collection.schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                vector_fields.append({
                    'name': field.name,
                    'dim': field.params.get('dim', 0) if hasattr(field, 'params') else 0
                })
        
        logger.info(f"找到向量字段: {vector_fields}")
        
        if not vector_fields:
            logger.error("集合中没有找到向量字段")
            return False
        
        # 测试每个向量字段
        search_success = False
        for vector_field in vector_fields:
            field_name = vector_field['name']
            dim = vector_field['dim']
            
            if dim == 0:
                logger.warning(f"向量字段 {field_name} 维度未知，跳过")
                continue
            
            logger.info(f"\n测试向量字段: {field_name} (维度: {dim})")
            
            try:
                # 创建测试向量
                test_vector = [0.1 * i for i in range(dim)]
                
                # 创建匹配表达式
                match_expr = MatchDenseExpr(
                    vector_column_name=field_name,
                    embedding_data=test_vector,
                    embedding_data_type="float",
                    distance_type="L2",
                    topn=10
                )
                
                # 执行搜索
                logger.info("执行搜索...")
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
                
                # 检查结果
                total = conn.getTotal(results)
                chunk_ids = conn.getChunkIds(results)
                
                logger.info(f"搜索结果:")
                logger.info(f"  - 总数: {total}")
                logger.info(f"  - 返回条目: {len(chunk_ids)}")
                logger.info(f"  - 结果格式: {type(results)}")
                
                if total > 0:
                    logger.info(f"  - 示例ID: {chunk_ids[:3]}")
                    logger.info(f"  ✓ 搜索成功! 找到 {total} 条结果")
                    search_success = True
                    
                    # 显示详细结果
                    if "hits" in results and "hits" in results["hits"]:
                        for i, hit in enumerate(results["hits"]["hits"][:3]):
                            logger.info(f"    结果 {i+1}:")
                            logger.info(f"      ID: {hit.get('_id', 'N/A')}")
                            logger.info(f"      Score: {hit.get('_score', 'N/A')}")
                            source = hit.get('_source', {})
                            logger.info(f"      Doc ID: {source.get('doc_id', 'N/A')}")
                            logger.info(f"      KB ID: {source.get('kb_id', 'N/A')}")
                            content = source.get('content_ltks', '')
                            if content and len(content) > 100:
                                content = content[:100] + "..."
                            logger.info(f"      Content: {content}")
                    
                    break
                else:
                    logger.warning(f"  搜索无结果")
                    
            except Exception as e:
                logger.error(f"  搜索失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        if search_success:
            logger.info("\n✓ Milvus 搜索修复验证成功!")
            return True
        else:
            logger.error("\n✗ Milvus 搜索修复验证失败 - 所有搜索都无结果")
            return False
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_pagination():
    """测试分页功能"""
    try:
        logger.info("\n=== 测试分页功能 ===")
        
        from rag.utils.milvus_conn import MilvusConnection
        from rag.utils.doc_store_conn import MatchDenseExpr, OrderByExpr
        from pymilvus import utility
        
        conn = MilvusConnection()
        collections = utility.list_collections(using=conn.alias)
        
        if not collections:
            logger.warning("没有集合可用于分页测试")
            return False
        
        # 找到有数据的集合
        test_collection_name = None
        for collection_name in collections:
            from pymilvus import Collection
            collection = Collection(name=collection_name, using=conn.alias)
            collection.load()
            if collection.num_entities > 0:
                test_collection_name = collection_name
                break
        
        if not test_collection_name:
            logger.warning("没有有数据的集合可用于分页测试")
            return False
        
        parts = test_collection_name.split('_')
        # 对于 ragflow_ 开头的集合，特殊处理
        if test_collection_name.startswith('ragflow_'):
            index_name = '_'.join(parts[:-1])
            kb_id = parts[-1]
        else:
            index_name = '_'.join(parts[:-1])
            kb_id = parts[-1]
        
        # 获取向量字段
        from pymilvus import Collection
        collection = Collection(name=test_collection_name, using=conn.alias)
        collection.load()
        
        from pymilvus import DataType
        vector_field = None
        for field in collection.schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                vector_field = field
                break
        
        if not vector_field:
            logger.warning("没有向量字段可用于分页测试")
            return False
        
        dim = vector_field.params.get('dim', 1024) if hasattr(vector_field, 'params') else 1024
        test_vector = [0.1 * i for i in range(dim)]
        
        match_expr = MatchDenseExpr(
            vector_column_name=vector_field.name,
            embedding_data=test_vector,
            embedding_data_type="float",
            distance_type="L2",
            topn=20
        )
        
        # 测试不同的分页参数
        page_tests = [
            {"offset": 0, "limit": 3},
            {"offset": 3, "limit": 3},
            {"offset": 6, "limit": 3},
        ]
        
        for test_params in page_tests:
            offset = test_params["offset"]
            limit = test_params["limit"]
            
            logger.info(f"测试分页: offset={offset}, limit={limit}")
            
            results = conn.search(
                selectFields=["id", "doc_id"],
                highlightFields=[],
                condition={},
                matchExprs=[match_expr],
                orderBy=OrderByExpr(),
                offset=offset,
                limit=limit,
                indexNames=[index_name],
                knowledgebaseIds=[kb_id]
            )
            
            total = conn.getTotal(results)
            chunk_ids = conn.getChunkIds(results)
            
            logger.info(f"  结果: 总数={total}, 返回={len(chunk_ids)}, IDs={chunk_ids}")
        
        logger.info("✓ 分页测试完成")
        return True
        
    except Exception as e:
        logger.error(f"分页测试失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始 Milvus 搜索修复验证...")
    
    # 测试搜索修复
    search_success = test_milvus_search_fix()
    
    # 测试分页功能
    pagination_success = test_pagination()
    
    if search_success and pagination_success:
        logger.info("\n🎉 所有测试通过! Milvus 搜索修复成功!")
    else:
        logger.error("\n❌ 部分测试失败，需要进一步调试")
    
    logger.info("测试完成")

if __name__ == "__main__":
    main()
