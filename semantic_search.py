# semantic_search.py
"""
基于 Milvus 的语义搜索演示
流程：
1. 从 CSV 读取向量数据
2. 将数据插入 Milvus
3. 对查询文本进行 embedding
4. 在 Milvus 中搜索语义相似的文本
"""

import csv
from typing import List, Dict, Tuple
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from embedding import get_embedding_model


class SemanticSearch:
    """语义搜索引擎（基于 Milvus）"""
    
    def __init__(
        self,
        milvus_url: str = "http://localhost:19530",
        collection_name: str = "semantic_search_demo",
        embedding_method: str = "sentence-transformers",
        embedding_kwargs: Dict = None
    ):
        """
        初始化语义搜索引擎
        
        Args:
            milvus_url: Milvus 服务器地址
            collection_name: Collection 名称
            embedding_method: embedding 方法
            embedding_kwargs: embedding 参数
        """
        self.milvus_url = milvus_url
        self.collection_name = collection_name
        self.client = MilvusClient(milvus_url)
        
        # 初始化 embedding 模型
        if embedding_kwargs is None:
            embedding_kwargs = {}
        self.embedder = get_embedding_model(embedding_method, **embedding_kwargs)
        self.embedding_dim = self.embedder.embedding_dim if hasattr(self.embedder, 'embedding_dim') else 384
        
        print(f"✓ 已初始化语义搜索引擎 (Collection: {collection_name})")
    
    def drop_collection_if_exists(self):
        """如果 collection 已存在，删除它"""
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                print(f"✓ 已删除旧 collection: {self.collection_name}")
        except Exception as e:
            print(f"  提示: {e}")
    
    def create_collection(self, force_recreate: bool = False):
        """
        创建 collection
        
        Args:
            force_recreate: 是否强制重建（删除旧数据）
        """
        if force_recreate:
            self.drop_collection_if_exists()
        
        # 定义 schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
        ]
        schema = CollectionSchema(fields=fields)
        
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            metric_type="COSINE"  # 使用余弦相似度
        )
        print(f"✓ 已创建 collection: {self.collection_name}")
        print(f"  向量维度: {self.embedding_dim}")
    
    def insert_from_csv(self, csv_file: str) -> int:
        """
        从 CSV 文件中读取向量数据并插入 Milvus
        
        Args:
            csv_file: CSV 文件路径（应包含 id, text, vector 三列）
            
        Returns:
            插入的数据条数
        """
        print(f"\n开始读取 CSV: {csv_file}")
        entities = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vector_str = row['vector']
                vector = [float(v) for v in vector_str.split(',')]
                
                entities.append({
                    'id': int(row['id']),
                    'text': row['text'],
                    'vector': vector
                })
        
        print(f"✓ 读取 {len(entities)} 条数据，开始插入...")
        self.client.insert(self.collection_name, entities)
        print(f"✓ 成功插入 {len(entities)} 条数据")
        
        return len(entities)
    
    def insert_texts(self, texts: List[str]) -> int:
        """
        直接插入文本列表（自动 embedding）
        
        Args:
            texts: 文本列表
            
        Returns:
            插入的数据条数
        """
        print(f"\n开始 embedding {len(texts)} 条文本...")
        vectors = self.embedder.embed_texts(texts)
        
        entities = [
            {
                'id': i,
                'text': text,
                'vector': vector
            }
            for i, (text, vector) in enumerate(zip(texts, vectors))
        ]
        
        print(f"开始插入数据...")
        self.client.insert(self.collection_name, entities)
        print(f"✓ 成功插入 {len(entities)} 条数据")
        
        return len(entities)
    
    def load_collection(self):
        """加载 collection 到内存（搜索前必须调用）"""
        self.client.load_collection(collection_name=self.collection_name)
        print(f"✓ 已加载 collection 到内存")
    
    def search(self, query_text: str, limit: int = 5) -> List[Tuple[str, float, int]]:
        """
        语义搜索
        
        Args:
            query_text: 查询文本
            limit: 返回结果数量
            
        Returns:
            [(文本, 相似度分数, id), ...] 的列表
        """
        # 对查询文本进行 embedding
        query_vector = self.embedder.embed_text(query_text)
        
        # 搜索
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 32}},
            limit=limit,
            output_fields=["id", "text"]
        )
        
        # 解析结果
        search_results = []
        for hit in results[0]:
            search_results.append((
                hit['entity']['text'],
                hit['distance'],
                hit['entity']['id']
            ))
        
        return search_results
    
    def release_collection(self):
        """释放 collection 内存"""
        self.client.release_collection(collection_name=self.collection_name)
        print(f"✓ 已释放 collection")
    
    def drop_collection(self):
        """删除 collection"""
        self.client.drop_collection(self.collection_name)
        print(f"✓ 已删除 collection: {self.collection_name}")


def run_semantic_search_demo(csv_file: str = None, texts: List[str] = None):
    """
    运行语义搜索演示
    
    Args:
        csv_file: CSV 文件路径（优先使用）
        texts: 文本列表（如果没有 CSV）
    """
    print("=" * 70)
    print("基于 Milvus 的语义搜索演示")
    print("=" * 70)
    
    # 如果没有指定文本，使用默认示例
    if csv_file is None and texts is None:
        texts = [
            "机器学习是人工智能的一个重要分支",
            "深度学习在计算机视觉中应用广泛",
            "自然语言处理技术在文本分析中很有用",
            "神经网络模型性能不断提升",
            "数据科学是当今重要的技能",
            "向量数据库用于存储高维向量",
            "Milvus 是一个开源向量数据库",
            "语义搜索基于文本相似度",
            "Transformer 模型改变了 NLP 领域",
            "BERT 是预训练的语言模型"
        ]
    
    # 初始化搜索引擎
    search_engine = SemanticSearch(
        collection_name="semantic_search_demo",
        embedding_method="sentence-transformers",
        embedding_kwargs={"model_name": "all-MiniLM-L6-v2"}
    )
    
    # 创建 collection
    search_engine.create_collection(force_recreate=True)
    
    # 插入数据
    if csv_file:
        search_engine.insert_from_csv(csv_file)
    else:
        search_engine.insert_texts(texts)
    
    # 加载到内存
    search_engine.load_collection()
    
    # 语义搜索测试
    print("\n" + "=" * 70)
    print("语义搜索测试")
    print("=" * 70)
    
    test_queries = [
        "人工智能技术",
        "向量数据库应用",
        "深度学习模型",
        "NLP 技术"
    ]
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        results = search_engine.search(query, limit=3)
        
        for rank, (text, score, doc_id) in enumerate(results, 1):
            print(f"  {rank}. [ID:{doc_id}] (相似度: {score:.4f})")
            print(f"     {text}")
    
    # 清理
    search_engine.release_collection()
    # search_engine.drop_collection()  # 可选：删除 collection
    
    print("\n✓ 演示完成！")


if __name__ == "__main__":
    # 运行演示
    run_semantic_search_demo()
    
    # 如果有 CSV 文件，可以这样使用：
    # run_semantic_search_demo(csv_file="test_dataset.csv")
