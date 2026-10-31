"""Run the labeled retrieval cases and save ranked-page observations."""

import argparse
import csv
import re
from pathlib import Path

from .retrieval import (
    TextChunk,
    load_pdf,
    retrieve,
    retrieve_hybrid,
    retrieve_semantic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_cases.csv"


def load_evaluation_source(path: Path):
    if path.suffix.lower() != ".txt":
        return load_pdf(path)

    text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^--- PAGE (\d+) ---\s*$", text)
    return [
        TextChunk(source=path.name, page=int(sections[i]), text=sections[i + 1].strip())
        for i in range(1, len(sections), 2)
        if sections[i + 1].strip()
    ]


def pages_text(results):
    return ";".join(str(chunk.page) for chunk, _score in results)


def gold_rank(results, gold_pages):
    for rank, (chunk, _score) in enumerate(results, start=1):
        if chunk.page in gold_pages:
            return rank
    return ""


def evaluate(cases_path: Path = CASES_PATH):
    top_k = 20
    with cases_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        cases = list(reader)

    fieldnames = [
        "source_file",
        "question",
        "answerable",
        "gold_pdf_pages",
        "tfidf_top3_pages",
        "tfidf_hit_at_3",
        "semantic_top3_pages",
        "semantic_hit_at_3",
        "semantic_top20_pages",
        "semantic_gold_rank",
        "semantic_hit_at_20",
        "hybrid_top3_pages",
        "hybrid_gold_rank",
        "hybrid_hit_at_3",
    ]

    chunk_cache = {}
    for case_number, case in enumerate(cases, start=1):
        source = case["source_file"]
        if source not in chunk_cache:
            source_path = (
                PROJECT_ROOT / "data" / "evaluation" / source
                if source.lower().endswith(".txt")
                else PROJECT_ROOT / "data" / "documents" / source
            )
            if not source_path.is_file():
                raise FileNotFoundError(f"找不到评估资料：{source_path}")
            chunk_cache[source] = load_evaluation_source(source_path)

        chunks = chunk_cache[source]
        question = case["question"]
        tfidf_results = retrieve(question, chunks, top_k=top_k)
        semantic_results = retrieve_semantic(question, chunks, top_k=top_k)
        hybrid_results = retrieve_hybrid(question, chunks, top_k=top_k)
        gold_pages = {
            int(page) for page in case.get("gold_pdf_pages", "").split(";") if page
        }
        tfidf_rank = gold_rank(tfidf_results, gold_pages)
        semantic_rank = gold_rank(semantic_results, gold_pages)
        hybrid_rank = gold_rank(hybrid_results, gold_pages)
        tfidf_top3_rank = gold_rank(tfidf_results[:3], gold_pages)
        hybrid_top3_rank = gold_rank(hybrid_results[:3], gold_pages)

        case.update(
            {
                "tfidf_top3_pages": pages_text(tfidf_results[:3]),
                "tfidf_hit_at_3": str(bool(tfidf_top3_rank)).lower(),
                "semantic_top20_pages": pages_text(semantic_results),
                "semantic_gold_rank": semantic_rank,
                "semantic_hit_at_20": str(bool(semantic_rank)).lower(),
                "hybrid_top3_pages": pages_text(hybrid_results[:3]),
                "hybrid_gold_rank": hybrid_rank,
                "hybrid_hit_at_3": str(bool(hybrid_top3_rank)).lower(),
            }
        )
        print(
            f"{case_number}: tfidf rank={tfidf_rank or '未命中'}, "
            f"semantic rank={semantic_rank or '未命中'}, "
            f"hybrid rank={hybrid_rank or '未命中'}; "
            f"hybrid top3={case['hybrid_top3_pages']}"
        )

    with cases_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)

    answerable_cases = [case for case in cases if case["answerable"] == "true"]
    tfidf_hits3 = sum(
        case["tfidf_hit_at_3"] == "true" for case in answerable_cases
    )
    semantic_hits = sum(
        case["semantic_hit_at_20"] == "true" for case in answerable_cases
    )
    hybrid_hits3 = sum(
        case["hybrid_hit_at_3"] == "true" for case in answerable_cases
    )
    hybrid_hits20 = sum(
        bool(case["hybrid_gold_rank"]) for case in answerable_cases
    )
    count = len(answerable_cases)
    print(
        f"Saved {count} cases. TF-IDF Hit@3={tfidf_hits3}/{count}; "
        f"semantic Hit@{top_k}={semantic_hits}/{count}; "
        f"hybrid Hit@3={hybrid_hits3}/{count}; "
        f"hybrid Hit@{top_k}={hybrid_hits20}/{count}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    args = parser.parse_args()
    evaluate(args.cases)
