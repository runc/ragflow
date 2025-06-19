#!/usr/bin/env python3
"""
调试 /v1/chunk/list 接口的详细测试脚本
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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_chunk_list_api():
    """测试 chunk list API 的完整流程"""
    try:
        logger.info("=== 测试 /v1/chunk/list 接口 ===")
        
        # 导入必要的模块
        from api.db.services.document_service import DocumentService
        from api.db.services.knowledgebase_service import KnowledgebaseService
        from rag.nlp import search
        from api import settings
        
        # 测试参数
        doc_id = "a106ea544ce911f0ae6b5df10d5df26e"
        page = 1
        size = 10
        question = ""  # 空关键词搜索
        
        logger.info(f"测试文档ID: {doc_id}")
        logger.info(f"页码: {page}, 大小: {size}")
        logger.info(f"关键词: '{question}'")
        
        # 步骤1: 检查文档是否存在
        logger.info("\n=== 步骤1: 检查文档是否存在 ===")
        e, doc = DocumentService.get_by_id(doc_id)
        if not e:
            logger.error(f"文档 {doc_id} 不存在!")
            return False
        
        logger.info(f"✓ 文档存在: {doc.to_dict()}")
        
        # 步骤2: 获取 tenant_id
        logger.info("\n=== 步骤2: 获取 tenant_id ===")
        tenant_id = DocumentService.get_tenant_id(doc_id)
        if not tenant_id:
            logger.error("无法获取 tenant_id!")
            return False
        
        logger.info(f"✓ tenant_id: {tenant_id}")
        
        # 步骤3: 获取知识库IDs
        logger.info("\n=== 步骤3: 获取知识库IDs ===")
        kb_ids = KnowledgebaseService.get_kb_ids(tenant_id)
        logger.info(f"✓ 知识库IDs: {kb_ids}")
        
        # 步骤4: 构建搜索查询
        logger.info("\n=== 步骤4: 构建搜索查询 ===")
        query = {
            "doc_ids": [doc_id], 
            "page": page, 
            "size": size, 
            "question": question, 
            "sort": True
        }
        logger.info(f"搜索查询: {query}")
        
        # 步骤5: 初始化 settings
        logger.info("\n=== 步骤5: 初始化 settings ===")
        settings.init_settings()
        logger.info(f"retrievaler 初始化状态: {settings.retrievaler is not None}")
        
        # 步骤6: 执行搜索
        logger.info("\n=== 步骤6: 执行搜索 ===")
        index_name = search.index_name(tenant_id)
        logger.info(f"索引名: {index_name}")
        
        sres = settings.retrievaler.search(query, index_name, kb_ids, highlight=True)
        logger.info(f"搜索结果总数: {sres.total}")
        logger.info(f"返回的chunk IDs: {sres.ids}")
        
        # 步骤7: 构建响应数据
        logger.info("\n=== 步骤7: 构建响应数据 ===")
        res = {"total": sres.total, "chunks": [], "doc": doc.to_dict()}
        
        if sres.total > 0:
            logger.info("✓ 找到了chunks，开始构建响应数据...")
            for id in sres.ids:
                logger.debug(f"处理chunk ID: {id}")
                logger.debug(f"字段数据: {sres.field.get(id, {})}")
                
                d = {
                    "chunk_id": id,
                    "content_with_weight": sres.field[id].get("content_with_weight", ""),
                    "doc_id": sres.field[id]["doc_id"],
                    "docnm_kwd": sres.field[id]["docnm_kwd"],
                    "important_kwd": sres.field[id].get("important_kwd", []),
                    "question_kwd": sres.field[id].get("question_kwd", []),
                    "image_id": sres.field[id].get("img_id", ""),
                    "available_int": int(sres.field[id].get("available_int", 1)),
                    "positions": sres.field[id].get("position_int", []),
                }
                res["chunks"].append(d)
                logger.debug(f"添加chunk: {d}")
        else:
            logger.warning("❌ 没有找到任何chunks!")
        
        logger.info(f"\n=== 最终结果 ===")
        logger.info(f"总数: {res['total']}")
        logger.info(f"返回chunks数量: {len(res['chunks'])}")
        
        if res['chunks']:
            logger.info("✓ chunk list 测试成功!")
            for i, chunk in enumerate(res['chunks'][:3]):  # 只显示前3个
                logger.info(f"Chunk {i+1}: {chunk['chunk_id'][:16]}... (doc_id: {chunk['doc_id']})")
        else:
            logger.error("❌ chunk list 返回空结果!")
            
            # 额外调试：直接查询Milvus
            logger.info("\n=== 额外调试：直接查询Milvus ===")
            from rag.utils.milvus_conn import MilvusConnection
            from rag.utils.doc_store_conn import OrderByExpr
            
            conn = MilvusConnection()
            
            # 尝试直接搜索该文档的chunks
            direct_res = conn.search(
                selectFields=["id", "doc_id", "kb_id", "content_ltks"],
                highlightFields=[],
                condition={"doc_id": doc_id},
                matchExprs=[],
                orderBy=OrderByExpr(),
                offset=0,
                limit=10,
                indexNames=[index_name],
                knowledgebaseIds=kb_ids
            )
            
            total = conn.getTotal(direct_res)
            chunk_ids = conn.getChunkIds(direct_res)
            logger.info(f"直接Milvus查询结果 - 总数: {total}, IDs: {chunk_ids}")
            
            if total > 0:
                fields = conn.getFields(direct_res, ["id", "doc_id", "kb_id", "content_ltks"])
                logger.info(f"直接查询的字段数据: {fields}")
                
        return res['total'] > 0
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    logger.info("开始测试 /v1/chunk/list 接口...")
    
    success = test_chunk_list_api()
    
    if success:
        logger.info("\n🎉 chunk list 接口测试成功!")
    else:
        logger.error("\n❌ chunk list 接口测试失败")
    
    logger.info("测试完成")

if __name__ == "__main__":
    main()
