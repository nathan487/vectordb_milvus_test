# dataset_to_csv.py
"""
将文本数据集转换为 CSV，包含原始文本和 embedding 向量
"""

import csv
import json
from typing import List, Dict
from embedding import get_embedding_model


def texts_to_csv(
    texts: List[str],
    output_csv: str,
    embedding_method: str = "sentence-transformers",
    embedding_kwargs: Dict = None,
    batch_size: int = 32,
    show_progress: bool = True
):
    """
    将文本列表转换为 CSV（包含向量）
    
    Args:
        texts: 文本列表
        output_csv: 输出 CSV 文件路径
        embedding_method: embedding 方法（"sentence-transformers", "transformers", "siliconflow"）
        embedding_kwargs: 传给 embedding 模型的参数字典
        batch_size: embedding 批处理大小
        show_progress: 是否显示进度
    """
    if embedding_kwargs is None:
        embedding_kwargs = {}
    
    print(f"开始 embedding {len(texts)} 条文本...")
    embedder = get_embedding_model(embedding_method, **embedding_kwargs)
    vectors = embedder.embed_texts(texts, batch_size=batch_size, show_progress_bar=show_progress)
    
    print(f"开始写入 CSV: {output_csv}")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'text', 'vector'])
        writer.writeheader()
        
        for idx, (text, vector) in enumerate(zip(texts, vectors)):
            vector_str = ','.join([str(v) for v in vector])
            writer.writerow({
                'id': idx,
                'text': text,
                'vector': vector_str
            })
    
    print(f"✓ CSV 写入完成: {output_csv}")
    print(f"  文本数: {len(texts)}")
    print(f"  向量维度: {len(vectors[0])}")


def json_to_csv(
    json_file: str,
    output_csv: str,
    text_field: str = "text",
    embedding_method: str = "sentence-transformers",
    embedding_kwargs: Dict = None
):
    """
    从 JSON 文件中读取文本，转换为 CSV
    
    Args:
        json_file: JSON 文件路径（格式: [{"text": "..."}, ...] 或 {"data": [...]})
        output_csv: 输出 CSV 文件路径
        text_field: JSON 中文本字段的名称
        embedding_method: embedding 方法
        embedding_kwargs: embedding 参数
    """
    print(f"读取 JSON 文件: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理不同的 JSON 格式
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and 'data' in data:
        items = data['data']
    else:
        raise ValueError("JSON 格式不支持，应为列表或包含 'data' 键的字典")
    
    texts = [item.get(text_field, "") for item in items]
    texts_to_csv(texts, output_csv, embedding_method, embedding_kwargs)


def txt_to_csv(
    txt_file: str,
    output_csv: str,
    embedding_method: str = "sentence-transformers",
    embedding_kwargs: Dict = None
):
    """
    从文本文件中读取（每行一条文本），转换为 CSV
    
    Args:
        txt_file: 文本文件路径
        output_csv: 输出 CSV 文件路径
        embedding_method: embedding 方法
        embedding_kwargs: embedding 参数
    """
    print(f"读取文本文件: {txt_file}")
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"读取 {len(texts)} 条文本")
    texts_to_csv(texts, output_csv, embedding_method, embedding_kwargs)


if __name__ == "__main__":
    # 示例：创建测试数据集并转为 CSV
    
    test_texts = [
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
    
    print("=" * 60)
    print("文本数据集转换为 CSV")
    print("=" * 60)
    
    # 转换为 CSV
    texts_to_csv(
        test_texts,
        "test_dataset.csv",
        embedding_method="sentence-transformers",
        embedding_kwargs={"model_name": "all-MiniLM-L6-v2"},
        show_progress=True
    )
    
    print("\n✓ 数据集转换完成！")
    print("  输出文件: test_dataset.csv")
    print("\n使用该 CSV 文件：")
    print("  1. 读取 CSV 并插入 Milvus")
    print("  2. 进行语义搜索测试")
