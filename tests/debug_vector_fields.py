#!/usr/bin/env python3
"""
调试向量字段检测的脚本
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

def debug_vector_field_detection():
    """调试向量字段检测"""
    try:
        from pymilvus import connections, utility, Collection, DataType
        
        # 连接 Milvus
        connections.connect(alias="debug_test", host="localhost", port="19530")
        
        # 获取集合
        collections_list = utility.list_collections(using="debug_test")
        
        for collection_name in collections_list:
            collection = Collection(name=collection_name, using="debug_test")
            collection.load()
            
            logger.info(f"\n=== 调试集合: {collection_name} ===")
            logger.info(f"实体数量: {collection.num_entities}")
            
            logger.info("字段详细信息:")
            for field in collection.schema.fields:
                logger.info(f"  字段名: {field.name}")
                logger.info(f"  数据类型: {field.dtype}")
                logger.info(f"  数据类型字符串: {str(field.dtype)}")
                logger.info(f"  是否为向量字段 (VECTOR in str): {'VECTOR' in str(field.dtype)}")
                logger.info(f"  是否为向量字段 (== FLOAT_VECTOR): {field.dtype == DataType.FLOAT_VECTOR}")
                logger.info(f"  是否为向量字段 (== BINARY_VECTOR): {field.dtype == DataType.BINARY_VECTOR}")
                
                if hasattr(field, 'params') and field.params:
                    logger.info(f"  参数: {field.params}")
                    if 'dim' in field.params:
                        logger.info(f"  维度: {field.params['dim']}")
                
                # 检查是否是向量字段的多种方法
                is_vector_1 = 'VECTOR' in str(field.dtype)
                is_vector_2 = field.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]
                is_vector_3 = field.dtype == DataType.FLOAT_VECTOR
                is_vector_4 = "_vec" in field.name
                
                logger.info(f"  向量检测方法1 ('VECTOR' in str): {is_vector_1}")
                logger.info(f"  向量检测方法2 (dtype in list): {is_vector_2}")
                logger.info(f"  向量检测方法3 (== FLOAT_VECTOR): {is_vector_3}")
                logger.info(f"  向量检测方法4 ('_vec' in name): {is_vector_4}")
                logger.info("  ---")
                
                # 如果是向量字段，尝试搜索
                if is_vector_2 and collection.num_entities > 0:
                    logger.info(f"  尝试搜索向量字段: {field.name}")
                    try:
                        dim = field.params.get('dim', 1024) if hasattr(field, 'params') else 1024
                        test_vector = [0.1 * i for i in range(dim)]
                        
                        search_results = collection.search(
                            data=[test_vector],
                            anns_field=field.name,
                            param={"metric_type": "L2", "params": {"nprobe": 10}},
                            limit=3,
                            output_fields=["id"]
                        )
                        
                        if search_results and len(search_results) > 0:
                            hits = search_results[0]
                            logger.info(f"    搜索成功: {len(hits)} 条结果")
                            for i, hit in enumerate(hits):
                                logger.info(f"      结果{i+1}: ID={hit.id}, 距离={hit.distance}")
                        else:
                            logger.warning(f"    搜索无结果")
                            
                    except Exception as e:
                        logger.error(f"    搜索失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"调试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    logger.info("开始向量字段检测调试...")
    debug_vector_field_detection()
    logger.info("调试完成")

if __name__ == "__main__":
    main()
