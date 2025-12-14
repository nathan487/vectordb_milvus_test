# quick_search.py
"""
快速语义搜索脚本
用法：
  python quick_search.py "查询文本"
  python quick_search.py "查询文本" -c collection_name -l 5
"""

import argparse
from pymilvus import MilvusClient
from embedding import get_embedding_model


def quick_search(
    query_text: str,
    collection_name: str = "semantic_search",
    milvus_url: str = "http://localhost:19530",
    limit: int = 5,
    embedding_method: str = "sentence-transformers",
    model_name: str = "all-MiniLM-L6-v2"
):
    """
    快速语义搜索
    
    Args:
        query_text: 查询文本
        collection_name: Collection 名称
        milvus_url: Milvus 地址
        limit: 返回结果数
        embedding_method: embedding 方法
        model_name: embedding 模型
    """
    print("=" * 70)
    print(f"快速语义搜索")
    print("=" * 70)
    
    # 初始化 embedding
    print(f"\n[1] 初始化 embedding 模型: {model_name}")
    try:
        embedder = get_embedding_model(
            embedding_method,
            model_name=model_name
        )
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False
    
    # 连接 Milvus
    print(f"\n[2] 连接 Milvus")
    try:
        client = MilvusClient(milvus_url)
        print(f"✓ 连接成功")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return False
    
    # 检查 collection
    print(f"\n[3] 检查 Collection: {collection_name}")
    try:
        if not client.has_collection(collection_name):
            print(f"✗ Collection 不存在: {collection_name}")
            return False
        print(f"✓ Collection 存在")
    except Exception as e:
        print(f"✗ 检查失败: {e}")
        return False
    
    # 对查询文本进行 embedding
    print(f"\n[4] 对查询文本进行 embedding")
    try:
        query_vector = embedder.embed_text(query_text)
        print(f"✓ Embedding 完成，维度: {len(query_vector)}")
    except Exception as e:
        print(f"✗ Embedding 失败: {e}")
        return False
    
    # 加载 collection
    print(f"\n[5] 加载 Collection 到内存")
    try:
        client.load_collection(collection_name)
        print(f"✓ 加载成功")
    except Exception as e:
        # collection 可能已加载，不需要报错
        print(f"  提示: {e}")
    
    # 执行搜索
    print(f"\n[6] 执行搜索 (limit={limit})")
    try:
        results = client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 32}},
            limit=limit,
            output_fields=["id", "text"]
        )
        print(f"✓ 搜索完成\n")
    except Exception as e:
        print(f"✗ 搜索失败: {e}")
        return False
    
    # 显示结果
    print("=" * 70)
    print(f"查询: '{query_text}'")
    print("=" * 70)
    
    if not results or not results[0]:
        print("未找到结果")
        return True
    
    for rank, hit in enumerate(results[0], 1):
        text = hit['entity']['text']
        score = hit['distance']
        doc_id = hit['entity']['id']
        
        print(f"\n{rank}. [ID:{doc_id}]")
        print(f"   相似度: {score:.4f}")
        print(f"   内容: {text}")
    
    print("\n" + "=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="快速语义搜索",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基础搜索
  python quick_search.py "人工智能"
  
  # 搜索并返回10个结果
  python quick_search.py "深度学习" -l 10
  
  # 使用自定义 collection
  python quick_search.py "向量数据库" -c my_collection
  
  # 使用中文模型
  python quick_search.py "机器学习" -m paraphrase-multilingual-MiniLM-L12-v2
        """
    )
    
    parser.add_argument(
        'query',
        type=str,
        help='查询文本'
    )
    
    parser.add_argument(
        '-c', '--collection',
        type=str,
        default='semantic_search',
        help='Collection 名称 (默认: semantic_search)'
    )
    
    parser.add_argument(
        '-l', '--limit',
        type=int,
        default=5,
        help='返回结果数 (默认: 5)'
    )
    
    parser.add_argument(
        '-u', '--url',
        type=str,
        default='http://localhost:19530',
        help='Milvus 服务器地址 (默认: http://localhost:19530)'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Embedding 模型 (默认: all-MiniLM-L6-v2)'
    )
    
    args = parser.parse_args()
    
    # 执行搜索
    success = quick_search(
        query_text=args.query,
        collection_name=args.collection,
        milvus_url=args.url,
        limit=args.limit,
        model_name=args.model
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
