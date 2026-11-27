"""Check an exported acceptance batch against the frozen reference labels."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check_results(path):
    reviews = read_csv(HERE / "reviews.csv")
    expected = read_csv(HERE / "expected.csv")
    actual = read_csv(path)
    if len(actual) != len(expected) or len(reviews) != len(expected):
        raise ValueError("Result row count differs from the reference.")
    categories = {}
    errors = []
    for index, (source, reference, result) in enumerate(zip(reviews, expected, actual), 1):
        if result["row"] != str(index) or result["text"] != source["text"]:
            raise ValueError(f"Source text or row order changed at row {index}.")
        passed = result["sentiment"] == reference["sentiment"] and result["status"] == reference["status"]
        counts = categories.setdefault(reference["category"], {"passed": 0, "total": 0})
        counts["total"] += 1
        counts["passed"] += passed
        if not passed:
            errors.append({"row": index, "category": reference["category"],
                           "expected": reference["sentiment"], "actual": result["sentiment"],
                           "status": result["status"]})
    identities = {(row["model"], row["model_digest"], row["ollama_version"]) for row in actual}
    if len(identities) != 1:
        raise ValueError("Results mix model or service identities.")
    model, digest, version = identities.pop()
    return {
        "model": model, "model_digest": digest, "ollama_version": version,
        "input_sha256": hashlib.sha256((HERE / "reviews.csv").read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256((HERE / "expected.csv").read_bytes()).hexdigest(),
        "rows": len(actual), "status_counts": dict(Counter(row["status"] for row in actual)),
        "review_correct": sum(value["passed"] for key, value in categories.items() if key != "empty"),
        "review_total": sum(value["total"] for key, value in categories.items() if key != "empty"),
        "categories": categories, "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(check_results(args.results), indent=2))


if __name__ == "__main__":
    main()
