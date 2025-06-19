#!/usr/bin/env python3
"""
验证 Milvus 中的 doc_id 字段值
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

def verify_milvus_doc_ids():
    """验证 Milvus 中的 doc_id 字段值"""
    try:
        logger.info("=== 验证 Milvus 中的 doc_id 字段值 ===")
        
        from rag.utils.milvus_conn import MilvusConnection
        from rag.utils.doc_store_conn import OrderByExpr
        from pymilvus import Collection
        
        # 创建连接
        conn = MilvusConnection()
        
        # 使用已知有数据的集合
        collection_name = "ragflow_7179adc24c1b11f0bb2a6b89a3fc27c2_7b5bf8b84c1b11f0bb2a6b89a3fc27c2"
        index_name = "ragflow_7179adc24c1b11f0bb2a6b89a3fc27c2"
        kb_id = "7b5bf8b84c1b11f0bb2a6b89a3fc27c2"
        target_doc_id = "a106ea544ce911f0ae6b5df10d5df26e"
        
        logger.info(f"测试集合: {collection_name}")
        logger.info(f"目标文档ID: {target_doc_id}")
        
        # 获取集合信息
        collection = Collection(name=collection_name, using=conn.alias)
        collection.load()
        
        logger.info(f"集合实体数: {collection.num_entities}")
        
        # 1. 先获取所有 doc_id 值，看看实际有什么
        logger.info("\n=== 获取所有 doc_id 值 ===")
        all_docs_res = conn.search(
            selectFields=["id", "doc_id", "kb_id"],
            highlightFields=[],
            condition={},  # 无过滤条件
            matchExprs=[],
            orderBy=OrderByExpr(),
            offset=0,
            limit=50,  # 获取更多结果
            indexNames=[index_name],
            knowledgebaseIds=[kb_id]
        )
        
        total = conn.getTotal(all_docs_res)
        chunk_ids = conn.getChunkIds(all_docs_res)
        logger.info(f"无过滤条件搜索结果 - 总数: {total}, 返回数量: {len(chunk_ids)}")
        
        if total > 0:
            fields = conn.getFields(all_docs_res, ["id", "doc_id", "kb_id"])
            logger.info("前10个文档的doc_id值:")
            doc_id_counts = {}
            for i, (chunk_id, data) in enumerate(fields.items()):
                if i < 10:
                    logger.info(f"  Chunk {i+1}: id={chunk_id}, doc_id='{data.get('doc_id', 'N/A')}', kb_id='{data.get('kb_id', 'N/A')}'")
                
                # 统计doc_id
                doc_id_val = data.get('doc_id', '')
                doc_id_counts[doc_id_val] = doc_id_counts.get(doc_id_val, 0) + 1
            
            logger.info(f"\ndoc_id 统计:")
            for doc_id, count in doc_id_counts.items():
                logger.info(f"  '{doc_id}': {count} chunks")
                if doc_id == target_doc_id:
                    logger.info(f"  ✓ 找到目标文档ID: {target_doc_id}")
        
        # 2. 尝试使用过滤条件搜索
        logger.info(f"\n=== 测试过滤条件搜索 ===")
        logger.info(f"目标doc_id: '{target_doc_id}'")
        
        # 测试不同的过滤条件格式
        test_conditions = [
            {"doc_id": target_doc_id},
            {"doc_id": [target_doc_id]},
        ]
        
        for i, condition in enumerate(test_conditions):
            logger.info(f"\n--- 测试条件 {i+1}: {condition} ---")
            filtered_res = conn.search(
                selectFields=["id", "doc_id", "kb_id"],
                highlightFields=[],
                condition=condition,
                matchExprs=[],
                orderBy=OrderByExpr(),
                offset=0,
                limit=10,
                indexNames=[index_name],
                knowledgebaseIds=[kb_id]
            )
            
            total = conn.getTotal(filtered_res)
            chunk_ids = conn.getChunkIds(filtered_res)
            logger.info(f"过滤搜索结果 - 总数: {total}, 返回数量: {len(chunk_ids)}")
            
            if total > 0:
                fields = conn.getFields(filtered_res, ["id", "doc_id", "kb_id"])
                logger.info("匹配的结果:")
                for chunk_id, data in fields.items():
                    logger.info(f"  id={chunk_id}, doc_id='{data.get('doc_id', 'N/A')}'")
            else:
                logger.warning("  ❌ 没有找到匹配结果")
        
        return total > 0
        
    except Exception as e:
        logger.error(f"验证失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    logger.info("开始验证 Milvus 中的 doc_id 字段值...")
    
    success = verify_milvus_doc_ids()
    
    if success:
        logger.info("\n🎉 验证完成!")
    else:
        logger.error("\n❌ 验证失败")
    
    logger.info("验证完成")

if __name__ == "__main__":
    main()
