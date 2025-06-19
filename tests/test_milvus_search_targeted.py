#!/usr/bin/env python3
"""
针对有数据集合的 Milvus 搜索测试脚本
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

def test_with_data_collection():
    """测试有数据的集合"""
    try:
        logger.info("=== 测试有数据的集合 ===")
        
        from rag.utils.milvus_conn import MilvusConnection
        from rag.utils.doc_store_conn import MatchDenseExpr, OrderByExpr
        from pymilvus import utility, Collection
        
        # 创建连接
        conn = MilvusConnection()
        
        # 获取所有集合
        collections = utility.list_collections(using=conn.alias)
        
        # 找到有数据的集合
        data_collection = None
        for collection_name in collections:
            collection = Collection(name=collection_name, using=conn.alias)
            collection.load()
            if collection.num_entities > 0:
                data_collection = collection_name
                logger.info(f"找到有数据的集合: {collection_name} (实体数: {collection.num_entities})")
                break
        
        if not data_collection:
            logger.error("没有找到有数据的集合")
            return False
        
        # 解析集合名称
        parts = data_collection.split('_')
        if len(parts) < 2:
            logger.error(f"集合名称格式不正确: {data_collection}")
            return False
        
        # 对于 ragflow_ 开头的集合，需要特殊处理
        if data_collection.startswith('ragflow_'):
            # ragflow_7179adc24c1b11f0bb2a6b89a3fc27c2_7b5bf8b84c1b11f0bb2a6b89a3fc27c2
            parts = data_collection.split('_')
            if len(parts) >= 3:
                index_name = '_'.join(parts[:-1])  # ragflow_7179adc24c1b11f0bb2a6b89a3fc27c2
                kb_id = parts[-1]  # 7b5bf8b84c1b11f0bb2a6b89a3fc27c2
            else:
                index_name = '_'.join(parts[:-1])
                kb_id = parts[-1]
        else:
            index_name = '_'.join(parts[:-1])
            kb_id = parts[-1]
        
        logger.info(f"解析结果 - 索引名: {index_name}, 知识库ID: {kb_id}")
        
        # 获取集合详细信息
        collection = Collection(name=data_collection, using=conn.alias)
        collection.load()
        
        # 找到向量字段
        vector_fields = []
        for field in collection.schema.fields:
            if 'VECTOR' in str(field.dtype):
                dim = field.params.get('dim', 0) if hasattr(field, 'params') else 0
                vector_fields.append({
                    'name': field.name,
                    'dim': dim
                })
        
        logger.info(f"找到向量字段: {vector_fields}")
        
        if not vector_fields:
            logger.error("集合中没有找到向量字段")
            return False
        
        # 测试每个向量字段
        for vector_field in vector_fields:
            field_name = vector_field['name']
            dim = vector_field['dim']
            
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
                logger.info(f"  - 结果类型: {type(results)}")
                
                if total > 0:
                    logger.info(f"  - 示例ID: {chunk_ids[:3]}")
                    logger.info(f"  ✓ 搜索成功! 找到 {total} 条结果")
                    
                    # 显示详细结果
                    if isinstance(results, dict) and "hits" in results:
                        hits_data = results["hits"]
                        if isinstance(hits_data, dict) and "hits" in hits_data:
                            for i, hit in enumerate(hits_data["hits"][:3]):
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
                    
                    return True
                else:
                    logger.warning(f"  搜索无结果")
                    
            except Exception as e:
                logger.error(f"  搜索失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        return False
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_direct_milvus_search():
    """直接使用 Milvus API 测试搜索"""
    try:
        logger.info("\n=== 直接 Milvus API 搜索测试 ===")
        
        from pymilvus import connections, utility, Collection
        
        # 连接 Milvus
        connections.connect(alias="direct_test", host="localhost", port="19530")
        
        # 找到有数据的集合
        collections_list = utility.list_collections(using="direct_test")
        data_collection_name = None
        
        for collection_name in collections_list:
            collection = Collection(name=collection_name, using="direct_test")
            collection.load()
            if collection.num_entities > 0:
                data_collection_name = collection_name
                break
        
        if not data_collection_name:
            logger.error("没有找到有数据的集合")
            return False
        
        logger.info(f"使用集合: {data_collection_name}")
        
        collection = Collection(name=data_collection_name, using="direct_test")
        collection.load()
        
        # 找到向量字段
        vector_field = None
        for field in collection.schema.fields:
            if 'VECTOR' in str(field.dtype):
                vector_field = field
                break
        
        if not vector_field:
            logger.error("没有找到向量字段")
            return False
        
        dim = vector_field.params.get('dim', 1024) if hasattr(vector_field, 'params') else 1024
        logger.info(f"向量字段: {vector_field.name}, 维度: {dim}")
        
        # 创建测试向量
        test_vector = [0.1 * i for i in range(dim)]
        
        # 执行搜索
        logger.info("执行直接搜索...")
        search_results = collection.search(
            data=[test_vector],
            anns_field=vector_field.name,
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=5,
            output_fields=["id", "doc_id", "kb_id"]
        )
        
        if search_results and len(search_results) > 0:
            hits = search_results[0]
            logger.info(f"直接搜索结果: {len(hits)} 条")
            for i, hit in enumerate(hits[:3]):
                logger.info(f"  结果 {i+1}: 距离={hit.distance}, ID={hit.id}")
                entity_dict = hit.entity.to_dict()
                logger.info(f"    数据: {entity_dict}")
            return True
        else:
            logger.warning("直接搜索无结果")
            return False
            
    except Exception as e:
        logger.error(f"直接搜索测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    logger.info("开始针对有数据集合的 Milvus 搜索测试...")
    
    # 1. 测试有数据的集合
    ragflow_success = test_with_data_collection()
    
    # 2. 直接 Milvus API 测试
    direct_success = test_direct_milvus_search()
    
    if ragflow_success:
        logger.info("\n🎉 RAGFlow 搜索测试成功!")
    else:
        logger.error("\n❌ RAGFlow 搜索测试失败")
    
    if direct_success:
        logger.info("🎉 直接 Milvus API 搜索测试成功!")
    else:
        logger.error("❌ 直接 Milvus API 搜索测试失败")
    
    logger.info("测试完成")

if __name__ == "__main__":
    main()
