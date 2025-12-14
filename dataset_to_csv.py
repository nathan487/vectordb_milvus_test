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


def main():
    """主函数：处理命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="将文本数据集转换为包含向量的 CSV 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 从文本文件转换
  python dataset_to_csv.py -t data.txt -o output.csv
  
  # 从 JSON 文件转换
  python dataset_to_csv.py --json data.json -o output.csv
  
  # 使用中文模型
  python dataset_to_csv.py -t data.txt -o output.csv -m paraphrase-multilingual-MiniLM-L12-v2
  
  # 自定义 batch_size 加速
  python dataset_to_csv.py -t data.txt -o output.csv -b 64
        """
    )
    
    parser.add_argument(
        '-t', '--txt',
        type=str,
        default='data.txt',
        help='输入的文本文件（每行一条文本，默认: data.txt）'
    )
    
    parser.add_argument(
        '-j', '--json',
        type=str,
        help='输入的 JSON 文件（如指定，则使用 JSON 而非文本文件）'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output.csv',
        help='输出的 CSV 文件路径（默认: output.csv）'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Embedding 模型（默认: all-MiniLM-L6-v2）'
    )
    
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=32,
        help='Embedding batch size（默认: 32，可增大加速）'
    )
    
    parser.add_argument(
        '--text-field',
        type=str,
        default='text',
        help='JSON 中文本字段的名称（仅当使用 --json 时有效，默认: text）'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("文本数据集转换为 CSV")
    print("=" * 70)
    
    # 选择转换方式
    if args.json:
        print(f"\n[输入] JSON 文件: {args.json}")
        print(f"[字段] 文本字段: {args.text_field}")
        json_to_csv(
            args.json,
            args.output,
            text_field=args.text_field,
            embedding_method="sentence-transformers",
            embedding_kwargs={"model_name": args.model}
        )
    else:
        print(f"\n[输入] 文本文件: {args.txt}")
        txt_to_csv(
            args.txt,
            args.output,
            embedding_method="sentence-transformers",
            embedding_kwargs={"model_name": args.model}
        )
    
    print("\n[配置]")
    print(f"  Embedding 模型: {args.model}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"[输出] CSV 文件: {args.output}")
    
    print("\n✓ 转换完成！")
    print("\n后续步骤:")
    print(f"  1. 导入到 Milvus: python import_csv_to_milvus.py -f {args.output}")
    print(f"  2. 进行搜索: python quick_search.py '查询文本'")


if __name__ == "__main__":
    main()
