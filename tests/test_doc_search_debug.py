#!/usr/bin/env python3
"""
专门测试查找特定 doc_id 的脚本
"""

import os
import sys
import logging

# 设置环境变量
os.environ['DOC_ENGINE'] = 'milvus'

# 添加项目路径
sys.path.insert(0, '/root/gitlab/runc/ragflow')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_specific_doc_search():
    """测试查找特定文档"""
    try:
        logger.info("=== 测试查找特定文档 ===")
        
        from pymilvus import connections, utility, Collection
        from api import settings
        from rag.nlp import search
        
        # 初始化设置
        settings.init_settings()
        
        # 目标参数
        doc_id = "a106ea544ce911f0ae6b5df10d5df26e"
        tenant_id = "7179adc24c1b11f0bb2a6b89a3fc27c2"
        kb_ids = ['0c140afc4c2211f08d6863b911b48031', '7b5bf8b84c1b11f0bb2a6b89a3fc27c2']
        
        logger.info(f"查找文档: {doc_id}")
        logger.info(f"租户ID: {tenant_id}")
        logger.info(f"知识库IDs: {kb_ids}")
        
        # 连接 Milvus
        connections.connect(
            alias="test",
            host="localhost",
            port="19530"
        )
        
        # 检查每个知识库对应的集合
        for kb_id in kb_ids:
            collection_name = f"ragflow_{tenant_id}_{kb_id}"
            logger.info(f"\n检查集合: {collection_name}")
            
            # 检查集合是否存在
            if not utility.has_collection(collection_name, using="test"):
                logger.warning(f"集合 {collection_name} 不存在")
                continue
            
            # 获取集合
            collection = Collection(name=collection_name, using="test")
            collection.load()
            
            logger.info(f"集合实体数: {collection.num_entities}")
            
            # 查询该集合中是否有指定的 doc_id
            try:
                # 查询前10条记录的 doc_id
                results = collection.query(
                    expr="",  # 空表达式匹配所有记录
                    output_fields=["doc_id", "id", "docnm_kwd"],
                    limit=10
                )
                
                logger.info(f"查询到 {len(results)} 条记录:")
                found_target = False
                for result in results:
                    logger.info(f"  ID: {result.get('id', 'N/A')}, doc_id: {result.get('doc_id', 'N/A')}, docnm_kwd: {result.get('docnm_kwd', 'N/A')}")
                    if result.get('doc_id') == doc_id:
                        found_target = True
                        logger.info(f"  ✓ 找到目标文档!")
                
                if not found_target:
                    # 尝试直接查询该 doc_id
                    logger.info(f"尝试直接查询 doc_id = {doc_id}")
                    specific_results = collection.query(
                        expr=f"doc_id == '{doc_id}'",
                        output_fields=["doc_id", "id", "docnm_kwd", "content_with_weight"],
                        limit=100
                    )
                    
                    if specific_results:
                        logger.info(f"直接查询找到 {len(specific_results)} 条记录:")
                        for result in specific_results:
                            logger.info(f"  ID: {result.get('id', 'N/A')}")
                            logger.info(f"  doc_id: {result.get('doc_id', 'N/A')}")
                            logger.info(f"  docnm_kwd: {result.get('docnm_kwd', 'N/A')}")
                            content = result.get('content_with_weight', '')
                            logger.info(f"  content: {content[:100]}..." if len(content) > 100 else f"  content: {content}")
                    else:
                        logger.info("直接查询也没有找到该文档")
                        
            except Exception as e:
                logger.error(f"查询集合 {collection_name} 时出错: {e}")
        
        logger.info("\n=== 使用 RAGFlow 检索器测试 ===")
        
        # 使用 RAGFlow 的检索器
        if settings.retrievaler:
            logger.info("retrievaler 已初始化")
            
            # 测试不同的查询
            queries = [
                {"doc_ids": [doc_id], "page": 1, "size": 10, "question": "", "sort": True},
                {"page": 1, "size": 10, "question": "", "sort": True},  # 不指定 doc_id
                {"doc_ids": [doc_id], "page": 1, "size": 10, "question": "test"},  # 有 question
            ]
            
            for i, query in enumerate(queries):
                logger.info(f"\n查询 {i+1}: {query}")
                try:
                    sres = settings.retrievaler.search(query, search.index_name(tenant_id), kb_ids, highlight=True)
                    logger.info(f"结果: total={sres.total}, ids={sres.ids[:3] if sres.ids else 'None'}")
                    
                    if sres.total > 0 and sres.ids:
                        sample_id = sres.ids[0]
                        if sample_id in sres.field:
                            sample_doc_id = sres.field[sample_id].get('doc_id', 'unknown')
                            logger.info(f"样本记录的 doc_id: {sample_doc_id}")
                except Exception as e:
                    logger.error(f"查询失败: {e}")
        else:
            logger.error("retrievaler 未初始化")
            
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_specific_doc_search()
