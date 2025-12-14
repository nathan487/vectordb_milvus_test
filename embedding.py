# embedding.py
"""
文本向量化工具模块
支持多种 embedding 模型：
- sentence-transformers（推荐，离线本地模型）
- transformers（Hugging Face Transformers）
- SiliconFlow API（云端服务）
"""

from typing import List, Union
import numpy as np


class EmbeddingBase:
    """Embedding 基类"""
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转为向量列表
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表，每个向量是浮点数列表
        """
        raise NotImplementedError
    
    def embed_text(self, text: str) -> List[float]:
        """将单条文本转为向量"""
        return self.embed_texts([text])[0]


class SentenceTransformersEmbedding(EmbeddingBase):
    """
    使用 sentence-transformers 的 embedding
    推荐使用，本地离线模型，速度快，效果好
    
    安装：pip install sentence-transformers
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化 embedding 模型
        
        Args:
            model_name: HuggingFace 模型名称
                常用模型：
                - "all-MiniLM-L6-v2"（英文，384维）
                - "all-mpnet-base-v2"（英文，768维，效果更好但更慢）
                - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"（多语言）
                - "sentence-transformers/distiluse-base-multilingual-cased-v2"（多语言）
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✓ 已加载 SentenceTransformers 模型: {model_name} (维度: {self.embedding_dim})")
        except ImportError:
            raise ImportError("请先安装 sentence-transformers: pip install sentence-transformers")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> List[List[float]]:
        """
        批量文本向量化
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小（默认32）
            show_progress_bar: 是否显示进度条
            
        Returns:
            向量列表
        """
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        return embeddings.tolist()


class TransformersEmbedding(EmbeddingBase):
    """
    使用 transformers + 手动平均池化的 embedding
    更灵活，但需要自己处理 token 和池化
    
    安装：pip install transformers torch
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        初始化模型
        
        Args:
            model_name: HuggingFace 模型名称
                如 "bert-base-uncased", "bert-base-chinese" 等
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.model_name = model_name
            self.torch = torch
            print(f"✓ 已加载 Transformers 模型: {model_name} (设备: {self.device})")
        except ImportError:
            raise ImportError("请先安装 transformers 和 torch: pip install transformers torch")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """文本向量化"""
        embeddings = []
        
        with self.torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # 使用 [CLS] token 的输出或平均池化
            last_hidden_state = outputs.last_hidden_state
            embeddings = last_hidden_state[:, 0, :].cpu().numpy().tolist()  # 取 [CLS] token
        
        return embeddings


class SiliconFlowEmbedding(EmbeddingBase):
    """
    使用 SiliconFlow API 的 embedding
    云端服务，无需本地部署，支持多种模型
    
    安装：pip install requests
    需要：API密钥（https://cloud.siliconflow.cn）
    """
    
    def __init__(self, api_key: str, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        初始化 SiliconFlow embedding
        
        Args:
            api_key: SiliconFlow API 密钥
            model_name: 模型名称
                常用模型：
                - "BAAI/bge-small-zh-v1.5"（中文，512维）
                - "BAAI/bge-large-zh-v1.5"（中文，1024维，效果更好）
                - "BAAI/bge-small-en-v1.5"（英文）
                - "BAAI/bge-large-en-v1.5"（英文）
        """
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("请先安装 requests: pip install requests")
        
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.siliconflow.cn/v1/embeddings"
        print(f"✓ 已初始化 SiliconFlow embedding (模型: {model_name})")
    
    def embed_texts(self, texts: List[str], batch_size: int = 25) -> List[List[float]]:
        """
        批量文本向量化（SiliconFlow 通常限制单次请求条数）
        
        Args:
            texts: 文本列表
            batch_size: 单次请求的文本数（默认25，避免超限）
            
        Returns:
            向量列表
        """
        embeddings = []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            payload = {
                "model": self.model_name,
                "input": batch
            }
            
            response = self.requests.post(self.api_url, json=payload, headers=headers)
            
            if response.status_code != 200:
                raise RuntimeError(f"SiliconFlow API 错误: {response.text}")
            
            data = response.json()
            batch_embeddings = [item["embedding"] for item in data["data"]]
            embeddings.extend(batch_embeddings)
        
        return embeddings


# 便捷函数：自动选择 embedding 方法
def get_embedding_model(method: str = "sentence-transformers", **kwargs) -> EmbeddingBase:
    """
    获取 embedding 模型
    
    Args:
        method: embedding 方法
            - "sentence-transformers"（推荐）
            - "transformers"
            - "siliconflow"
        **kwargs: 传给模型的其他参数
        
    Returns:
        EmbeddingBase 实例
        
    Example:
        # 方法1：sentence-transformers（推荐）
        embedder = get_embedding_model("sentence-transformers", model_name="all-MiniLM-L6-v2")
        vectors = embedder.embed_texts(["你好", "世界"])
        
        # 方法2：SiliconFlow
        embedder = get_embedding_model("siliconflow", api_key="your_api_key")
        vectors = embedder.embed_texts(["你好", "世界"])
    """
    if method == "sentence-transformers":
        return SentenceTransformersEmbedding(**kwargs)
    elif method == "transformers":
        return TransformersEmbedding(**kwargs)
    elif method == "siliconflow":
        return SiliconFlowEmbedding(**kwargs)
    else:
        raise ValueError(f"未知的 embedding 方法: {method}")


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Embedding 工具测试")
    print("=" * 60)
    
    # 示例文本
    test_texts = [
        "我喜欢吃苹果",
        "我喜欢吃香蕉",
        "天气很好",
        "今天天气怎么样"
    ]
    
    # 方法1：使用 sentence-transformers（推荐）
    print("\n[1] 使用 SentenceTransformers")
    try:
        embedder = get_embedding_model(
            "sentence-transformers",
            model_name="all-MiniLM-L6-v2"
        )
        vectors = embedder.embed_texts(test_texts)
        print(f"✓ 成功生成 {len(vectors)} 个向量")
        print(f"  向量维度: {len(vectors[0])}")
        print(f"  第一个向量前5个值: {vectors[0][:5]}")
    except Exception as e:
        print(f"✗ 错误: {e}")
    
    # 方法2：使用 SiliconFlow（需要 API 密钥）
    print("\n[2] 使用 SiliconFlow API")
    print("  需要 SiliconFlow API 密钥，可从 https://cloud.siliconflow.cn 获取")
    print("  使用方法:")
    print('    embedder = get_embedding_model("siliconflow", api_key="your_api_key")')
    print('    vectors = embedder.embed_texts(["你好", "世界"])')
