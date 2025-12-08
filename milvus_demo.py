# milvus_demo.py
import time
import numpy as np
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

COLLECTION = "secondexample"
DIM = 128
N = 200  # 插入向量数量（小规模示例）

def random_vectors(n, dim):
    return (np.random.random((n, dim)).astype(np.float32)).tolist()

def main():
    # 1. 连接 Milvus (确保 Milvus 已启动)
    client = MilvusClient("http://localhost:19530")
    print("Connected to Milvus.")

    # 2. 如果 collection 已存在，先删除（演示用）
    try:
        if client.has_collection(COLLECTION):
            print(f"Collection '{COLLECTION}' exists — dropping it first.")
            client.drop_collection(COLLECTION)
    except Exception:
        # has_collection 在部分版本可能不存在，忽略错误
        pass

    # 3. 创建 collection，指定主键字段 id（使用 FieldSchema 和 CollectionSchema）
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM)
    ]
    schema = CollectionSchema(fields=fields)
    client.create_collection(collection_name=COLLECTION, schema=schema, metric_type="L2")
    print(f"Created collection '{COLLECTION}', dim={DIM}.")

    # 4. 生成并插入随机向量数据
    vectors = random_vectors(N, DIM)
    # 插入时提供 id 字段
    entities = [{"id": i, "vector": v} for i, v in enumerate(vectors)]
    insert_result = client.insert(COLLECTION, entities)
    print(f"Inserted {len(insert_result)} entities.")

    # 5. 创建索引（HNSW 示例）
    # 先用 prepare_index_params 构造索引参数（兼容各版本）
    idx_params = client.prepare_index_params()
    idx_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="L2",
        params={"M": 16, "efConstruction": 200}
    )
    client.create_index(collection_name=COLLECTION, index_params=idx_params)
    print("Index (HNSW) creation requested.")

    # 等待短暂时间让索引开始构建（真实生产请用 describe_index / wait 方法）
    time.sleep(2)

    # 6. Load collection 到内存，准备搜索
    client.load_collection(collection_name=COLLECTION)
    print("Collection loaded into memory.")

    # 7. 做一次示例检索（用第一条向量做查询）
    q_vec = vectors[0]
    search_params = {"ef": 50}  # HNSW 搜索参数
    results = client.search(
        collection_name=COLLECTION,
        data=[q_vec],        # 支持批量查询：这里为单条
        limit=5,
        search_params=search_params
    )
    print("Search results (format may vary by pymilvus version):")
    for u in results:
        print(u)

    # 8. 清理：释放 collection（注释掉 drop 保留数据）
    client.release_collection(collection_name=COLLECTION)
    # client.drop_collection(COLLECTION)  # 取消注释可以删除 collection
    print("Demo finished. Collection released (data persisted).")

if __name__ == "__main__":
    main()
