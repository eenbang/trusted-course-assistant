import argparse
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
# TfidfVector 把文字转换成便于比较的数字特征
from sklearn.feature_extraction.text import TfidfVectorizer
# cosine_similarity 根据数字特征计算两段文字的相似程度
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class TextChunk:
    source: str
    page: int
    text: str


def load_pdf(path: Path) -> list[TextChunk]:
    """逐页提取 PDF 文本，并切成有重叠的片段。 返回结果是一组 TextChunk 对象"""
    reader = PdfReader(str(path))
    chunks = []

    chunk_size = 600
    overlap = 80
    step = chunk_size - overlap

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if not text:
            continue

        for start in range(0, len(text), step):
            piece = text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    TextChunk(
                        source=path.name,
                        page=page_number,
                        text=piece,
                    )
                )

            if start + chunk_size >= len(text):
                break

    return chunks


def retrieve(question: str, chunks: list[TextChunk], top_k: int = 3):
    """用中文字符 n-gram TF-IDF 找出最相关的文本片段。"""
    if not chunks:
        return []
    # 创建 TF-IDF向量器

    # analyzer=“char” 表示按字符分析
    # ngram_range=(2,4) 表示连续的2，3，4个字符组合
    vectorizer = TfidfVectorizer(
        analyzer="char",  
        ngram_range=(2, 4),
    )
    # 把所有片段转换成数字向量。
    # fit_transform 会先学习语料里的字符组合，再把每段文字转成向量。
    document_vectors = vectorizer.fit_transform(
        [chunk.text for chunk in chunks]
    )
    # 把用户的问题也转换成向量。
    # 这里使用 transform，因为向量器已经通过上一步学习过字符组合。
    question_vector = vectorizer.transform([question])
    # 计算问题向量和每个片段向量之间的余弦相似度
    scores = cosine_similarity(question_vector, document_vectors).ravel()

    ranked_indices = scores.argsort()[::-1][:top_k]

    return [
        (chunks[index], float(scores[index]))
        for index in ranked_indices
        if scores[index] > 0
    ]


def main():
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser(
        description="从课程 PDF 中检索与问题相关的片段"
    )
    # 为了方便测试
    # 第一个参数：PDF路径
    parser.add_argument("pdf", type=Path, help="PDF 文件路径")
    # 第二个参数：查询的问题
    parser.add_argument("question", help="要检索的问题")
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"找不到 PDF 文件：{args.pdf}")

    # 读取 PDF 并切分文本
    chunks = load_pdf(args.pdf)

    if not chunks:
        raise SystemExit(
            "没有提取到文本。这个 PDF 可能是扫描图片，之后需要增加 OCR。"
        )

    results = retrieve(args.question, chunks)

    if not results:
        print("没有找到明显相关的片段。")
        return

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\n[{rank}] 相似度：{score:.3f}")
        print(f"来源：{chunk.source}，第 {chunk.page} 页")
        print(chunk.text)


if __name__ == "__main__":
    main()