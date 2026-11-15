#!/usr/bin/env python3
"""Retrieval quality evaluation for the RAG API.

Reads a JSONL question set, calls ``POST /search`` on the embedding API, and
reports recall@k, MRR@10, and nDCG@10. See ``eval/README.md`` for the dataset
format and recommendations.

Usage:
    python scripts/eval_retrieval.py --questions eval/questions.jsonl
    python scripts/eval_retrieval.py --questions eval/questions.jsonl \
        --api-url http://localhost:8000 --top-k 50
    python scripts/eval_retrieval.py --questions eval/questions.jsonl \
        --json-out eval/baseline.json

The script uses only the standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_QUESTIONS = Path("eval/questions.jsonl")
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_TOP_K = 50
DEFAULT_RECALL_K = "5,10"

_ALLOWED_EXPECTED_FIELDS = {"file", "chunk", "text_contains"}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def post_json(url: str, payload: Dict[str, Any], timeout: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def _fail(where: str, message: str) -> None:
    raise SystemExit(f"{where}: {message}")


def _validate_expected(entry: Any, where: str) -> None:
    if not isinstance(entry, dict):
        _fail(where, "'relevant' entries must be JSON objects")
    unknown = set(entry) - _ALLOWED_EXPECTED_FIELDS
    if unknown:
        _fail(where, f"unknown relevant field(s): {', '.join(sorted(unknown))}")
    if not entry:
        _fail(where, "a 'relevant' entry must have at least one field")

    if "file" in entry and not isinstance(entry["file"], str):
        _fail(where, "'file' must be a string")
    if "text_contains" in entry and not isinstance(entry["text_contains"], str):
        _fail(where, "'text_contains' must be a string")
    if "chunk" in entry:
        try:
            chunk = int(entry["chunk"])
        except (TypeError, ValueError):
            _fail(where, "'chunk' must be an integer")
        if chunk < 0:
            _fail(where, "'chunk' must be >= 0")
        entry["chunk"] = chunk


def load_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        _fail(
            str(path),
            "question file not found. Copy eval/questions.example.jsonl and "
            "replace the examples with real questions.",
        )

    questions: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            where = f"{path}:{line_number}"
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(where, f"invalid JSON: {exc}")

            if not isinstance(item, dict):
                _fail(where, "each line must be a JSON object")
            question = item.get("question")
            relevant = item.get("relevant")
            if not isinstance(question, str) or not question.strip():
                _fail(where, "'question' must be a non-empty string")
            if not isinstance(relevant, list) or not relevant:
                _fail(where, "'relevant' must be a non-empty array")
            for entry in relevant:
                _validate_expected(entry, where)

            question_id = item.get("id")
            if question_id is None:
                question_id = f"q{len(questions) + 1:03d}"
            questions.append(
                {
                    "id": str(question_id),
                    "question": question.strip(),
                    "relevant": relevant,
                }
            )

    if not questions:
        _fail(str(path), "no questions found")
    return questions


# --------------------------------------------------------------------------- #
# Matching and metrics
# --------------------------------------------------------------------------- #


def _same_file(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    # Accept a host-side path (data/documents/x.txt) and a container-side path
    # (/data/documents/x.txt) as the same document.
    try:
        return Path(actual).name == Path(expected).name
    except (OSError, ValueError):
        return False


def _matches(hit: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    if "file" in expected and not _same_file(str(hit.get("file", "")), expected["file"]):
        return False
    if "chunk" in expected:
        try:
            actual_chunk = int(hit.get("chunk"))
        except (TypeError, ValueError):
            return False
        if actual_chunk != expected["chunk"]:
            return False
    if "text_contains" in expected:
        haystack = str(hit.get("text", "")).casefold()
        if expected["text_contains"].casefold() not in haystack:
            return False
    return True


def _first_relevant_rank(hits: List[Any], relevant: List[Dict[str, Any]]) -> Optional[int]:
    for rank, hit in enumerate(hits, 1):
        if isinstance(hit, dict) and any(_matches(hit, expected) for expected in relevant):
            return rank
    return None


def _ndcg_at_k(hits: List[Any], relevant: List[Dict[str, Any]], k: int) -> float:
    dcg = 0.0
    for rank, hit in enumerate(hits[:k], 1):
        if isinstance(hit, dict) and any(_matches(hit, expected) for expected in relevant):
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate(
    api_url: str,
    questions: List[Dict[str, Any]],
    top_k: int,
    recall_ks: List[int],
    timeout: float,
) -> List[Dict[str, Any]]:
    search_url = f"{api_url.rstrip('/')}/search"
    results: List[Dict[str, Any]] = []

    for question in questions:
        hits = post_json(
            search_url,
            {"query": question["question"], "top_k": top_k, "min_score": None},
            timeout,
        )
        if not isinstance(hits, list):
            raise RuntimeError(
                f"{question['id']}: /search returned {type(hits).__name__}, expected a list"
            )

        first_rank = _first_relevant_rank(hits, question["relevant"])
        result: Dict[str, Any] = {
            "id": question["id"],
            "question": question["question"],
            "expected": question["relevant"],
            "returned": len(hits),
            "first_rank": first_rank,
        }
        for k in recall_ks:
            result[f"recall@{k}"] = 1.0 if first_rank is not None and first_rank <= k else 0.0
        result["mrr@10"] = (
            1.0 / first_rank if first_rank is not None and first_rank <= 10 else 0.0
        )
        result["ndcg@10"] = _ndcg_at_k(hits, question["relevant"], 10)
        results.append(result)

    return results


def summarise(results: List[Dict[str, Any]], recall_ks: List[int], top_k: int) -> Dict[str, Any]:
    total = len(results)
    summary: Dict[str, Any] = {"questions": total, "top_k": top_k}
    for k in recall_ks:
        summary[f"recall@{k}"] = sum(result[f"recall@{k}"] for result in results) / total
    summary["mrr@10"] = sum(result["mrr@10"] for result in results) / total
    summary["ndcg@10"] = sum(result["ndcg@10"] for result in results) / total
    return summary


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_report(
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
    recall_ks: List[int],
    show_failures: int,
) -> None:
    print(f"Questions: {summary['questions']}  (top_k={summary['top_k']})")
    for k in recall_ks:
        print(f"  recall@{k}: {summary[f'recall@{k}']:.3f}")
    print(f"  MRR@10:   {summary['mrr@10']:.3f}")
    print(f"  nDCG@10:  {summary['ndcg@10']:.3f}")

    max_k = max(recall_ks)
    failures = [
        result
        for result in results
        if result["first_rank"] is None or result["first_rank"] > max_k
    ]
    print(f"Failures in top {max_k}: {len(failures)}/{len(results)}")

    if show_failures and failures:
        print()
        for result in failures[: max(0, show_failures)]:
            rank = result["first_rank"] if result["first_rank"] is not None else "-"
            print(f"  {result['id']}: rank={rank}  {result['question'][:100]}")
            print(f"    expected: {json.dumps(result['expected'], ensure_ascii=False)}")


def _parse_ks(value: str) -> List[int]:
    try:
        ks = sorted({int(part) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --recall-k: {value}") from exc
    if not ks or ks[0] < 1:
        raise argparse.ArgumentTypeError("--recall-k must contain positive integers")
    return ks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--recall-k", type=_parse_ks, default=_parse_ks(DEFAULT_RECALL_K))
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--show-failures", type=int, default=20)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--min-recall", type=float, default=None,
                        help="exit 1 if recall@max(recall-k) is below this value")
    parser.add_argument("--min-mrr", type=float, default=None,
                        help="exit 1 if MRR@10 is below this value")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate the question file and exit without calling the API")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    questions = load_questions(Path(args.questions))
    if args.limit is not None:
        questions = questions[: max(0, args.limit)]
        if not questions:
            _fail(str(args.questions), "--limit removed all questions")

    if args.validate_only:
        print(f"{len(questions)} question(s) OK: {args.questions}")
        return 0

    try:
        results = evaluate(
            api_url=args.api_url,
            questions=questions,
            top_k=args.top_k,
            recall_ks=args.recall_k,
            timeout=args.timeout,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read()[:500].decode("utf-8", "replace")
        print(f"HTTP {exc.code} from {args.api_url}: {body}")
        return 1
    except urllib.error.URLError as exc:
        print(f"Cannot reach {args.api_url}: {exc.reason}")
        return 1
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        print(f"Evaluation failed: {exc}")
        return 1

    summary = summarise(results, args.recall_k, args.top_k)
    print_report(summary, results, args.recall_k, args.show_failures)

    if args.json_out:
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(Path(args.questions)),
            "api_url": args.api_url,
            **summary,
            "results": results,
        }
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")

    max_k = max(args.recall_k)
    failed = False
    if args.min_recall is not None and summary[f"recall@{max_k}"] < args.min_recall:
        print(
            f"recall@{max_k}={summary[f'recall@{max_k}']:.3f} "
            f"is below --min-recall={args.min_recall:.3f}"
        )
        failed = True
    if args.min_mrr is not None and summary["mrr@10"] < args.min_mrr:
        print(f"MRR@10={summary['mrr@10']:.3f} is below --min-mrr={args.min_mrr:.3f}")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
