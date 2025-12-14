# README_SEMANTIC_SEARCH.md
# 基于 Milvus 的语义搜索工具

这个工具集提供了完整的文本向量化和语义搜索流程。

## 文件说明

### 1. `embedding.py` - Embedding 工具模块
提供多种文本向量化的实现方式。

**支持的方法：**
- `sentence-transformers`（推荐）：本地离线模型，速度快，效果好
- `transformers`：使用 HuggingFace Transformers，更灵活
- `siliconflow`：云端 API 服务，无需本地部署

**快速使用：**
```python
from embedding import get_embedding_model

# 方法1：SentenceTransformers（推荐，英文）
embedder = get_embedding_model("sentence-transformers", model_name="all-MiniLM-L6-v2")
vectors = embedder.embed_texts(["你好", "世界"])

# 方法2：SentenceTransformers（中文）
embedder = get_embedding_model(
    "sentence-transformers",
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 方法3：SiliconFlow API
embedder = get_embedding_model("siliconflow", api_key="your_api_key")
```

### 2. `dataset_to_csv.py` - 数据集处理脚本
将文本数据转换为包含向量的 CSV 文件。

**主要函数：**
- `texts_to_csv()`：文本列表 → CSV
- `json_to_csv()`：JSON 文件 → CSV
- `txt_to_csv()`：文本文件（每行一条）→ CSV

**使用示例：**
```python
from dataset_to_csv import texts_to_csv

texts = ["机器学习", "深度学习", "自然语言处理"]
texts_to_csv(texts, "output.csv")
```

### 3. `semantic_search.py` - 语义搜索引擎
基于 Milvus 的完整语义搜索系统。

**主要类：**
- `SemanticSearch`：语义搜索引擎

**使用示例：**
```python
from semantic_search import SemanticSearch

# 初始化
search_engine = SemanticSearch()

# 创建 collection
search_engine.create_collection(force_recreate=True)

# 插入数据（方式1：直接文本）
texts = ["机器学习", "深度学习", "NLP"]
search_engine.insert_texts(texts)

# 或者插入数据（方式2：从 CSV）
search_engine.insert_from_csv("data.csv")

# 加载到内存
search_engine.load_collection()

# 语义搜索
results = search_engine.search("人工智能", limit=5)
for text, score, doc_id in results:
    print(f"{text} (相似度: {score:.4f})")
```

---

## 完整工作流

### 步骤1：准备环境和安装依赖

```bash
# 创建虚拟环境（可选）
conda create -n semantic_search python=3.10
conda activate semantic_search

# 安装依赖
pip install sentence-transformers pymilvus numpy pandas requests
```

### 步骤2：准备数据并转换为向量 CSV

**方案 A：直接用文本列表**
```python
# test_embedding.py
from dataset_to_csv import texts_to_csv

texts = [
    "机器学习是人工智能的一个重要分支",
    "深度学习在计算机视觉中应用广泛",
    "自然语言处理技术在文本分析中很有用",
    "神经网络模型性能不断提升",
    "数据科学是当今重要的技能",
    # 添加更多文本...
]

texts_to_csv(
    texts,
    "my_dataset.csv",
    embedding_method="sentence-transformers",
    embedding_kwargs={"model_name": "all-MiniLM-L6-v2"}
)

# 运行：python test_embedding.py
```

**方案 B：从 JSON 文件**
```python
from dataset_to_csv import json_to_csv

json_to_csv(
    "data.json",
    "my_dataset.csv",
    text_field="content"  # JSON 中文本字段的名称
)
```

**方案 C：从文本文件（每行一条文本）**
```python
from dataset_to_csv import txt_to_csv

txt_to_csv("data.txt", "my_dataset.csv")
```

### 步骤3：运行语义搜索

```python
# test_search.py
from semantic_search import SemanticSearch

# 初始化搜索引擎
search = SemanticSearch()

# 创建 collection
search.create_collection(force_recreate=True)

# 从 CSV 插入数据
search.insert_from_csv("my_dataset.csv")

# 加载到内存
search.load_collection()

# 执行搜索
query = "人工智能技术应用"
results = search.search(query, limit=5)

print(f"查询: {query}")
for text, score, doc_id in results:
    print(f"  [ID:{doc_id}] 相似度: {score:.4f} - {text}")

# 清理
search.release_collection()
```

运行：`python test_search.py`

---

## 推荐的 Embedding 模型

### 英文模型
| 模型名称 | 维度 | 推荐场景 | 下载大小 |
|---------|------|--------|--------|
| `all-MiniLM-L6-v2` | 384 | 快速、轻量 | 90 MB |
| `all-mpnet-base-v2` | 768 | 高效果 | 430 MB |
| `all-roberta-large-v1` | 1024 | 最佳效果 | 700 MB |

### 中文模型
| 模型名称 | 维度 | 推荐场景 |
|---------|------|--------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 多语言（轻量） |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 多语言（高效果） |
| `sentence-transformers/distiluse-base-multilingual-cased-v2` | 512 | 多语言（均衡） |

### 专业中文模型（SiliconFlow）
| 模型名称 | 维度 | 特点 |
|---------|------|------|
| `BAAI/bge-small-zh-v1.5` | 512 | 轻量、中文优化 |
| `BAAI/bge-large-zh-v1.5` | 1024 | 大模型、高精度 |

---

## 常见问题

### Q1: 如何使用中文 embedding？
```python
embedder = get_embedding_model(
    "sentence-transformers",
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
```

### Q2: 如何加速 embedding？
- 使用更小的模型（如 `all-MiniLM-L6-v2`）
- 增加 `batch_size`（如 64 或 128）
- 使用 GPU（CUDA）

### Q3: 内存不足怎么办？
- 分批处理数据
- 使用更小的模型
- 降低 `batch_size`

### Q4: 搜索精度不够怎么办？
- 使用更大的 embedding 模型
- 调整 Milvus 搜索参数（如 `ef`, `nprobe`）
- 增加训练数据质量

---

## 性能优化建议

1. **首次运行**：模型会自动下载，耗时较长
2. **批量 embedding**：使用 `batch_size=32` 或更大
3. **向量数据库**：大数据量（>10k）推荐使用 GPU 版 Milvus
4. **索引类型**：`IVF_FLAT` 适合百万级，`HNSW` 适合千万级

---

## 相关资源

- [Sentence Transformers 官网](https://www.sbert.net/)
- [Milvus 官网](https://milvus.io/)
- [SiliconFlow 官网](https://cloud.siliconflow.cn/)
- [HuggingFace Models](https://huggingface.co/models)

---

## 许可证

MIT
