import argparse
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from pypdf import PdfReader
# TfidfVector 把文字转换成便于比较的数字特征
from sklearn.feature_extraction.text import TfidfVectorizer
# cosine_similarity 根据数字特征计算两段文字的相似程度
from sklearn.metrics.pairwise import cosine_similarity

from functools import lru_cache
from sentence_transformers import SentenceTransformer

@dataclass
class TextChunk:
    source: str
    page: int
    text: str


# 分块设置。以后调整这些值时，缓存会自动重新生成。
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

def get_pdf_fingerprint(path: Path) -> str:
    """根据文件路径、大小和修改时间判断文件是否变化。"""
    file_info = path.stat()

    identity = (
        f"{path.resolve()}|"
        f"{file_info.st_size}|"
        f"{file_info.st_mtime_ns}"
    )

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()

def load_pdf(path: Path) -> list[TextChunk]:
    """读取 PDF 并切分文本；如果已有匹配的缓存，则直接复用。"""
    # 缓存放在项目根目录的 data/indexes 文件夹中。
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "data" / "indexes"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 文件内容或分块设置改变时，会得到新的缓存文件名。
    pdf_fingerprint = get_pdf_fingerprint(path)
    cache_key = hashlib.sha256(
        f"{pdf_fingerprint}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"chunks_{cache_key}.json"

    # 如果缓存存在且可读取，就直接恢复文本片段，不再解析 PDF 页面。
    if cache_path.exists():
        try:
            cache_data = json.loads(cache_path.read_text(encoding="utf-8"))

            if cache_data["cache_key"] == cache_key:
                chunks = [
                    TextChunk(
                        source=path.name,
                        page=item["page"],
                        text=item["text"],
                    )
                    for item in cache_data["chunks"]
                ]
                print("[缓存] 已复用 PDF 文本片段")
                return chunks

        except (OSError, ValueError, KeyError, TypeError):
            # 缓存损坏或格式不匹配时，继续从 PDF 重新生成。
            pass

    print("[缓存] 首次解析 PDF 并生成文本片段")

    # 缓存不存在时，才真正打开 PDF 并提取文字。
    reader = PdfReader(str(path))
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if not text:
            continue

        for start in range(0, len(text), step):
            piece = text[start : start + CHUNK_SIZE].strip()

            if piece:
                chunks.append(
                    TextChunk(
                        source=path.name,
                        page=page_number,
                        text=piece,
                    )
                )

            if start + CHUNK_SIZE >= len(text):
                break

    # 保存页码和文本片段。缓存里不保存 PDF 原文文件本身。
    cache_data = {
        "cache_key": cache_key,
        "chunks": [
            {"page": chunk.page, "text": chunk.text}
            for chunk in chunks
        ],
    }
    cache_path.write_text(
        json.dumps(cache_data, ensure_ascii=False),
        encoding="utf-8",
    )

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

@lru_cache(maxsize=1)
def get_embedding_model():
    """在当前 Python 进程中复用已经加载的模型。"""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_or_create_document_vectors(chunks: list[TextChunk]):
    """读取本地语义向量缓存；没有缓存时计算并保存。"""
    # 索引文件放在项目根目录的 data/indexes 文件夹中。
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "data" / "indexes"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 根据片段内容、页码、分块设置和模型名称生成缓存标识。
    # 其中任何一项变化，都会生成一份新缓存，避免误用旧向量。
    hasher = hashlib.sha256()
    hasher.update(
        f"{EMBEDDING_MODEL_NAME}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode("utf-8")
    )

    for chunk in chunks:
        hasher.update(str(chunk.page).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(chunk.text.encode("utf-8"))
        hasher.update(b"\0")

    cache_key = hasher.hexdigest()
    cache_path = cache_dir / f"vectors_{cache_key}.npz"

    # 如果本地已经有对应向量，就直接读取。
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                saved_key = cache["cache_key"].item()
                vectors = cache["vectors"]

                # 检查缓存标识和向量数量，避免读到不匹配的缓存。
                if saved_key == cache_key and vectors.shape[0] == len(chunks):
                    print("[缓存] 已复用语义向量")
                    return vectors.copy()

        except (OSError, ValueError, KeyError, TypeError, EOFError):
            # 缓存文件损坏或格式不匹配时，重新计算。
            pass

    print("[缓存] 首次计算文本片段的语义向量")

    # 仅在缓存不存在时，才对所有资料片段计算向量。
    model = get_embedding_model()
    vectors = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # 保存缓存标识和向量。allow_pickle=False 可避免读取任意 Python 对象。
    np.savez_compressed(
        cache_path,
        cache_key=np.array(cache_key),
        vectors=vectors.astype(np.float32),
    )

    return vectors


def retrieve_semantic(
    question: str,
    chunks: list[TextChunk],
    top_k: int = 3,
):
    """用缓存的资料向量检索最相关的文本片段。"""
    if not chunks:
        return []

    model = get_embedding_model()

    # 资料向量从本地缓存读取；只有首次运行时才会计算。
    document_vectors = get_or_create_document_vectors(chunks)

    # 将问题变成向量。BGE 中文检索模型使用这段查询提示语。
    instruction = "为这个句子生成表示以用于检索相关文章："
    question_vector = model.encode(
        [instruction + question],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    # 比较问题向量与资料向量，并按相似度从高到低排序。
    scores = cosine_similarity(question_vector, document_vectors).ravel()
    ranked_indices = scores.argsort()[::-1][:top_k]

    return [
        (chunks[index], float(scores[index]))
        for index in ranked_indices
    ]


def retrieve_hybrid(
    question: str,
    chunks: list[TextChunk],
    top_k: int = 3,
    rrf_k: int = 60,
):
    """Fuse lexical and semantic rankings with Reciprocal Rank Fusion."""
    if not chunks:
        return []

    # Use full rankings so a candidate highly ranked by either retriever is kept.
    lexical = retrieve(question, chunks, top_k=len(chunks))
    semantic = retrieve_semantic(question, chunks, top_k=len(chunks))
    fused_scores: dict[int, float] = {}
    chunk_indices = {id(chunk): index for index, chunk in enumerate(chunks)}

    # RRF uses ranks instead of raw scores, whose scales differ by method.
    for ranking in (lexical, semantic):
        for rank, (chunk, _score) in enumerate(ranking, start=1):
            index = chunk_indices[id(chunk)]
            fused_scores[index] = fused_scores.get(index, 0.0) + 1.0 / (rrf_k + rank)

    ranked_indices = sorted(
        fused_scores,
        key=lambda index: (-fused_scores[index], index),
    )[:top_k]
    return [
        (chunks[index], fused_scores[index])
        for index in ranked_indices
    ]


def main():
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser(
        description="从课程 PDF 中检索与问题相关的片段"
    )
    # 为了方便测试
    # 增加新参数
    parser.add_argument(
    "--method",
    choices=["tfidf", "semantic", "hybrid"],
    default="tfidf",
    help="选择检索方法：tfidf、semantic 或 hybrid",
    )   
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

    if args.method == "semantic":
        results = retrieve_semantic(args.question, chunks)
    elif args.method == "hybrid":
        results = retrieve_hybrid(args.question, chunks)
    else:
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
