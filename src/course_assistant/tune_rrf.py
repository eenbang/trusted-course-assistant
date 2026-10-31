"""Sweep RRF parameters on the labeled retrieval cases."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from .evaluate_retrieval import (
    CASES_PATH,
    PROJECT_ROOT,
    load_evaluation_source,
)
from .retrieval import (
    DEFAULT_RRF_K,
    DEFAULT_TFIDF_WEIGHT,
    TextChunk,
    fuse_rankings_with_rrf,
    retrieve,
    retrieve_semantic,
)


TOP_K = 20
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "rrf_tuning.csv"
)
RRF_K_VALUES = (5, 10, 20, 40, 60, 80, 100)
TFIDF_WEIGHT_VALUES = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SEMANTIC_WEIGHT = 1.0


@dataclass
class PreparedCase:
    case_number: int
    source: str
    question: str
    chunks: list[TextChunk]
    gold_pages: set[int]
    lexical: list[tuple[TextChunk, float]]
    semantic: list[tuple[TextChunk, float]]


def read_cases(cases_path: Path) -> list[dict[str, str]]:
    with cases_path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_case_source(source: str) -> list[TextChunk]:
    source_dir = (
        PROJECT_ROOT / "data" / "evaluation"
        if source.lower().endswith(".txt")
        else PROJECT_ROOT / "data" / "documents"
    )
    source_path = source_dir / source
    if not source_path.is_file():
        raise FileNotFoundError(f"找不到评估资料：{source_path}")
    return load_evaluation_source(source_path)


def prepare_cases(cases_path: Path) -> list[PreparedCase]:
    prepared = []
    chunk_cache: dict[str, list[TextChunk]] = {}
    cases = read_cases(cases_path)

    for case_number, case in enumerate(cases, start=1):
        if case["answerable"] != "true":
            continue

        source = case["source_file"]
        if source not in chunk_cache:
            chunk_cache[source] = load_case_source(source)
        chunks = chunk_cache[source]
        question = case["question"]

        print(f"准备案例 {case_number}/{len(cases)}：{question}")
        prepared.append(
            PreparedCase(
                case_number=case_number,
                source=source,
                question=question,
                chunks=chunks,
                gold_pages={
                    int(page)
                    for page in case.get("gold_pdf_pages", "").split(";")
                    if page
                },
                lexical=retrieve(question, chunks, top_k=len(chunks)),
                semantic=retrieve_semantic(
                    question,
                    chunks,
                    top_k=len(chunks),
                ),
            )
        )

    return prepared


def first_gold_rank(results, gold_pages: set[int]) -> int | None:
    for rank, (chunk, _score) in enumerate(results, start=1):
        if chunk.page in gold_pages:
            return rank
    return None


def evaluate_configuration(
    cases: list[PreparedCase],
    rrf_k: int,
    tfidf_weight: float,
    semantic_weight: float,
) -> dict[str, int | float]:
    ranks = []
    for case in cases:
        results = fuse_rankings_with_rrf(
            case.lexical,
            case.semantic,
            case.chunks,
            top_k=TOP_K,
            rrf_k=rrf_k,
            tfidf_weight=tfidf_weight,
            semantic_weight=semantic_weight,
        )
        ranks.append(first_gold_rank(results, case.gold_pages))

    total = len(cases)
    hit_at_3_count = sum(rank is not None and rank <= 3 for rank in ranks)
    hit_at_20_count = sum(rank is not None for rank in ranks)
    mrr_at_20 = sum(
        1.0 / rank for rank in ranks if rank is not None
    ) / total
    return {
        "rrf_k": rrf_k,
        "tfidf_weight": tfidf_weight,
        "semantic_weight": semantic_weight,
        "hit_at_3_count": hit_at_3_count,
        "hit_at_3": hit_at_3_count / total,
        "hit_at_20_count": hit_at_20_count,
        "hit_at_20": hit_at_20_count / total,
        "mrr_at_20": mrr_at_20,
        "case_count": total,
    }


def ranking_key(row: dict[str, int | float]) -> tuple[float, float, float, float]:
    """Prefer effectiveness, then settings close to the conventional default."""
    return (
        float(row["hit_at_3"]),
        float(row["hit_at_20"]),
        float(row["mrr_at_20"]),
        -(abs(float(row["tfidf_weight"]) - DEFAULT_TFIDF_WEIGHT) + abs(
            float(row["rrf_k"]) - DEFAULT_RRF_K
        ) / DEFAULT_RRF_K),
    )


def tune(cases_path: Path, output_path: Path) -> list[dict[str, int | float]]:
    cases = prepare_cases(cases_path)
    if not cases:
        raise ValueError("评估集中没有可回答案例，无法调整 RRF 参数")

    rows = []
    for rrf_k in RRF_K_VALUES:
        for tfidf_weight in TFIDF_WEIGHT_VALUES:
            rows.append(
                evaluate_configuration(
                    cases,
                    rrf_k=rrf_k,
                    tfidf_weight=tfidf_weight,
                    semantic_weight=SEMANTIC_WEIGHT,
                )
            )

    rows.sort(key=ranking_key, reverse=True)
    fieldnames = [
        "rrf_k",
        "tfidf_weight",
        "semantic_weight",
        "hit_at_3_count",
        "hit_at_3",
        "hit_at_20_count",
        "hit_at_20",
        "mrr_at_20",
        "case_count",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nRRF 参数扫描结果（按 Hit@3、Hit@20、MRR@20 排序）：")
    for row in rows[:10]:
        print(
            f"k={row['rrf_k']}, "
            f"tfidf_weight={row['tfidf_weight']:.2f}, "
            f"semantic_weight={row['semantic_weight']:.2f}, "
            f"Hit@3={row['hit_at_3_count']}/{row['case_count']}, "
            f"Hit@20={row['hit_at_20_count']}/{row['case_count']}, "
            f"MRR@20={row['mrr_at_20']:.4f}"
        )
    print(f"\n完整结果已写入：{output_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    tune(args.cases, args.output)
