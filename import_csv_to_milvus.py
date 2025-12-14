# import_csv_to_milvus.py
"""
将 CSV 文件中的向量数据导入到 Milvus
"""

import csv
import argparse
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType


def import_csv_to_milvus(
    csv_file: str,
    collection_name: str = "semantic_search",
    milvus_url: str = "http://localhost:19530",
    force_recreate: bool = False
):
    """
    将 CSV 文件导入到 Milvus
    
    Args:
        csv_file: CSV 文件路径（应包含 id, text, vector 三列）
        collection_name: Collection 名称
        milvus_url: Milvus 服务器地址
        force_recreate: 是否删除旧 collection 重新创建
    """
    print("=" * 70)
    print(f"Milvus CSV 导入工具")
    print("=" * 70)
    
    # 连接 Milvus
    print(f"\n[1] 连接 Milvus: {milvus_url}")
    client = MilvusClient(milvus_url)
    print("✓ 连接成功")
    
    # 处理旧 collection
    print(f"\n[2] 处理 Collection: {collection_name}")
    try:
        if client.has_collection(collection_name):
            if force_recreate:
                print(f"  旧 collection 存在，删除中...")
                client.drop_collection(collection_name)
                print(f"  ✓ 已删除旧 collection")
            else:
                print(f"  Collection 已存在，将追加数据")
    except Exception as e:
        print(f"  提示: {e}")
    
    # 读取 CSV 获取向量维度
    print(f"\n[3] 读取 CSV 文件: {csv_file}")
    entities = []
    embedding_dim = None
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            vector_str = row['vector']
            vector = [float(v) for v in vector_str.split(',')]
            
            # 获取向量维度（从第一行）
            if embedding_dim is None:
                embedding_dim = len(vector)
                print(f"  检测到向量维度: {embedding_dim}")
            
            entities.append({
                'id': int(row['id']),
                'text': row['text'],
                'vector': vector
            })
        
        num_records = len(entities)
        print(f"✓ 读取 {num_records} 条记录")
    
    # 创建 collection（如果不存在）
    if not client.has_collection(collection_name):
        print(f"\n[4] 创建 Collection")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
        ]
        schema = CollectionSchema(fields=fields)
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            metric_type="COSINE"
        )
        print(f"✓ 已创建 Collection: {collection_name}")
        print(f"  - 向量维度: {embedding_dim}")
        print(f"  - 距离度量: COSINE")
    else:
        print(f"\n[4] Collection 已存在，跳过创建")
    
    # 插入数据
    print(f"\n[5] 插入数据到 Milvus")
    try:
        result = client.insert(collection_name, entities)
        print(f"✓ 成功插入 {len(entities)} 条记录")
        print(f"  插入结果: {result}")
    except Exception as e:
        print(f"✗ 插入失败: {e}")
        return False
    
    # 创建索引
    print(f"\n[6] 创建索引")
    try:
        client.create_index(
            collection_name=collection_name,
            field_name="vector",
            index_type="HNSW",
            params={"M": 8, "efConstruction": 64}
        )
        print(f"✓ 索引创建成功")
    except Exception as e:
        print(f"  提示: 索引可能已存在或其他原因 - {e}")
    
    # 统计信息
    print(f"\n[7] Collection 统计信息")
    try:
        stats = client.describe_collection(collection_name)
        print(f"✓ Collection 详情:")
        print(f"  - 名称: {stats['name']}")
        print(f"  - 记录数: {stats.get('num_entities', 'N/A')}")
        print(f"  - 字段数: {len(stats['fields'])}")
    except Exception as e:
        print(f"  获取信息失败: {e}")
    
    print("\n" + "=" * 70)
    print("✓ 导入完成！")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="将 CSV 文件导入到 Milvus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 默认导入 test_dataset.csv
  python import_csv_to_milvus.py
  
  # 导入自定义 CSV 文件
  python import_csv_to_milvus.py -f my_data.csv
  
  # 导入并重建 collection
  python import_csv_to_milvus.py -f my_data.csv --force
  
  # 自定义 collection 名称
  python import_csv_to_milvus.py -f my_data.csv -c my_collection
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        default='test_dataset.csv',
        help='CSV 文件路径 (默认: test_dataset.csv)'
    )
    
    parser.add_argument(
        '-c', '--collection',
        type=str,
        default='semantic_search',
        help='Collection 名称 (默认: semantic_search)'
    )
    
    parser.add_argument(
        '-u', '--url',
        type=str,
        default='http://localhost:19530',
        help='Milvus 服务器地址 (默认: http://localhost:19530)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制删除旧 collection 重新创建'
    )
    
    args = parser.parse_args()
    
    # 执行导入
    success = import_csv_to_milvus(
        csv_file=args.file,
        collection_name=args.collection,
        milvus_url=args.url,
        force_recreate=args.force
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
