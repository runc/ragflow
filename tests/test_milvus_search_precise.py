#!/usr/bin/env python3
"""
精确诊断 RAGFlow Milvus 搜索问题的脚本
"""

import os
import sys
import logging

# 设置环境变量
os.environ['DOC_ENGINE'] = 'milvus'

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ragflow'))

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ragflow_search_precise():
    """精确测试 RAGFlow 搜索问题"""
    try:
        logger.info("=== 精确诊断 RAGFlow 搜索问题 ===")
        
        from rag.utils.milvus_conn import MilvusConnection
        from rag.utils.doc_store_conn import MatchDenseExpr, OrderByExpr
        from pymilvus import utility, Collection, DataType
        
        # 创建连接
        conn = MilvusConnection()
        
        # 使用已知有数据的集合
        collection_name = "ragflow_7179adc24c1b11f0bb2a6b89a3fc27c2_7b5bf8b84c1b11f0bb2a6b89a3fc27c2"
        index_name = "ragflow_7179adc24c1b11f0bb2a6b89a3fc27c2"
        kb_id = "7b5bf8b84c1b11f0bb2a6b89a3fc27c2"
        
        logger.info(f"测试集合: {collection_name}")
        logger.info(f"索引名: {index_name}")
        logger.info(f"知识库ID: {kb_id}")
        
        # 获取集合详细信息
        collection = Collection(name=collection_name, using=conn.alias)
        collection.load()
        
        logger.info(f"集合实体数: {collection.num_entities}")
        
        # 找到向量字段
        vector_field = None
        for field in collection.schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                vector_field = field
                break
        
        if not vector_field:
            logger.error("没有找到向量字段")
            return False
        
        dim = vector_field.params.get('dim', 1024)
        logger.info(f"向量字段: {vector_field.name}, 维度: {dim}")
        
        # 创建测试向量
        test_vector = [0.1 * i for i in range(dim)]
        
        # 创建匹配表达式
        match_expr = MatchDenseExpr(
            vector_column_name=vector_field.name,
            embedding_data=test_vector,
            embedding_data_type="float",
            distance_type="L2",
            topn=10
        )
        
        logger.info("\n=== 测试 RAGFlow 搜索 ===")
        
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
        
        logger.info(f"搜索结果类型: {type(results)}")
        logger.info(f"搜索结果: {results}")
        
        # 检查结果
        total = conn.getTotal(results)
        chunk_ids = conn.getChunkIds(results)
        
        logger.info(f"总数: {total}")
        logger.info(f"返回ID列表: {chunk_ids}")
        
        if total > 0:
            logger.info("✓ RAGFlow 搜索成功!")
            
            # 显示详细结果
            fields = conn.getFields(results, ["id", "doc_id", "kb_id", "content_ltks"])
            logger.info(f"字段数据: {fields}")
            
            return True
        else:
            logger.error("✗ RAGFlow 搜索返回空结果")
            return False
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    logger.info("开始精确诊断 RAGFlow Milvus 搜索问题...")
    
    success = test_ragflow_search_precise()
    
    if success:
        logger.info("\n🎉 RAGFlow 搜索测试成功!")
    else:
        logger.error("\n❌ RAGFlow 搜索测试失败")
    
    logger.info("诊断完成")

if __name__ == "__main__":
    main()
