"""Run the labeled retrieval cases and save ranked-page observations."""

import argparse
import csv
from pathlib import Path

from .retrieval import load_pdf, retrieve_hybrid, retrieve_semantic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_cases.csv"


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

    new_fields = [
        "semantic_top20_pages",
        "semantic_gold_rank",
        "semantic_hit_at_20",
        "hybrid_top3_pages",
        "hybrid_top20_pages",
        "hybrid_gold_rank",
        "hybrid_hit_at_3",
        "hybrid_hit_at_20",
    ]
    for field in new_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    chunk_cache = {}
    for case in cases:
        source = case["source_file"]
        if source not in chunk_cache:
            pdf_path = PROJECT_ROOT / "data" / "documents" / source
            if not pdf_path.is_file():
                raise FileNotFoundError(f"找不到评估资料：{pdf_path}")
            chunk_cache[source] = load_pdf(pdf_path)

        chunks = chunk_cache[source]
        question = case["question"]
        semantic_results = retrieve_semantic(question, chunks, top_k=top_k)
        hybrid_results = retrieve_hybrid(question, chunks, top_k=top_k)
        gold_pages = {
            int(page) for page in case.get("gold_pdf_pages", "").split(";") if page
        }
        semantic_rank = gold_rank(semantic_results, gold_pages)
        hybrid_rank = gold_rank(hybrid_results, gold_pages)
        hybrid_top3_rank = gold_rank(hybrid_results[:3], gold_pages)

        case.update(
            {
                "semantic_top20_pages": pages_text(semantic_results),
                "semantic_gold_rank": semantic_rank,
                "semantic_hit_at_20": str(bool(semantic_rank)).lower(),
                "hybrid_top3_pages": pages_text(hybrid_results[:3]),
                "hybrid_top20_pages": pages_text(hybrid_results),
                "hybrid_gold_rank": hybrid_rank,
                "hybrid_hit_at_3": str(bool(hybrid_top3_rank)).lower(),
                "hybrid_hit_at_20": str(bool(hybrid_rank)).lower(),
            }
        )
        print(
            f"{case['case_id']}: semantic rank={semantic_rank or '未命中'}, "
            f"hybrid rank={hybrid_rank or '未命中'}; "
            f"hybrid top3={case['hybrid_top3_pages']}"
        )

    with cases_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)

    semantic_hits = sum(case["semantic_hit_at_20"] == "true" for case in cases)
    hybrid_hits3 = sum(case["hybrid_hit_at_3"] == "true" for case in cases)
    hybrid_hits20 = sum(case["hybrid_hit_at_20"] == "true" for case in cases)
    count = len(cases)
    print(
        f"Saved {count} cases. Semantic Hit@{top_k}={semantic_hits}/{count}; "
        f"hybrid Hit@3={hybrid_hits3}/{count}; "
        f"hybrid Hit@{top_k}={hybrid_hits20}/{count}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    args = parser.parse_args()
    evaluate(args.cases)
