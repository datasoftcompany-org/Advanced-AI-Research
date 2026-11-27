"""Evaluate the approved binary prompt on non-neutral TweetEval sentiment posts."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path[:0] = [str(RESEARCH / name) for name in (
    "prompt-instruction-ablation", "paired-sentiment-evaluation", "local-sentiment-evaluation"
)]
from evaluate_ablation import text_key
from compare import align, run_model
from evaluate import LABELS, metrics, sha256


def build_sample(directory, protocol):
    data = {}
    for name, expected in protocol["file_hashes"].items():
        raw = (directory / name).read_bytes()
        if sha256(raw) != expected:
            raise ValueError(f"Source hash mismatch: {name}")
        data[name] = raw.decode("utf-8").splitlines()
    texts, labels = data["test_text.txt"], data["test_labels.txt"]
    if len(texts) != len(labels):
        raise ValueError("Text and label counts differ")
    if any(label not in ("0", "1", "2") for label in labels + data["train_labels.txt"]):
        raise ValueError("Unknown source label")
    groups = defaultdict(list)
    for index, (text, label) in enumerate(zip(texts, labels), 1):
        if not text.strip():
            raise ValueError("Empty source text")
        groups[text_key(text)].append((index, text, label))
    records = []
    for rows in groups.values():
        if len({row[2] for row in rows}) != 1:
            continue
        index, text, label = rows[0]
        if label != "1":
            records.append({"id": f"tweeteval:test:{index}", "text": text,
                            "label": {"0": "negative", "2": "positive"}[label], "group": "social"})
    selected = []
    for label in LABELS:
        rows = sorted((row for row in records if row["label"] == label),
                      key=lambda row: sha256(f'{protocol["salt"]}:{row["id"]}'.encode()))
        if len(rows) < protocol["per_label"]:
            raise ValueError("Insufficient source examples")
        selected.extend(rows[:protocol["per_label"]])
    return selected, {
        "file_hashes": protocol["file_hashes"],
        "original_test_counts": dict(Counter(labels)),
        "training_counts": dict(Counter(data["train_labels.txt"])),
        "eligible_binary_records": len(records),
        "test": [{"id": row["id"], "label": row["label"], "text_sha256": text_key(row["text"])}
                 for row in selected],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sample-only", action="store_true")
    args = parser.parse_args()
    raw = (HERE / "protocol.json").read_bytes()
    protocol = json.loads(raw)
    base_raw = (RESEARCH / "local-sentiment-evaluation/protocol.json").read_bytes()
    base = json.loads(base_raw)
    examples, sample = build_sample(args.data, protocol)
    sample.update({"protocol_sha256": sha256(raw), "base_protocol_sha256": sha256(base_raw)})
    if args.sample_only:
        print(json.dumps(sample, indent=2))
        return
    if sample != json.loads((HERE / "sample.json").read_bytes()):
        raise ValueError("Sample differs from frozen manifest")
    counts = sample["training_counts"]
    majority = max(LABELS, key=lambda label: (counts.get({"negative": "0", "positive": "2"}[label], 0),
                                              label == base["majority_tie_break"]))
    identities = {}
    rounds = []
    for order in (protocol["models"], list(reversed(protocol["models"]))):
        runs = {}
        for model in order:
            print(f"Evaluating {model}", file=sys.stderr, flush=True)
            run = run_model(model, examples, base, protocol["warmup"])
            identities.setdefault(model, run["identity"])
            if identities[model] != run["identity"]:
                raise ValueError("Model identity changed")
            runs[model] = run
        rounds.append({"order": order, "runs": runs})
    changed = {
        model: [a["id"] for a, b in align(run["rows"], rounds[1]["runs"][model]["rows"])
                if a["prediction"] != b["prediction"]]
        for model, run in rounds[0]["runs"].items()
    }
    print(json.dumps({
        "runtime": {"python": platform.python_version(), "architecture": platform.machine()},
        "sample": sample, "baseline_label": majority,
        "baseline": metrics([row | {"prediction": majority} for row in examples], "prediction"),
        "rounds": rounds, "changed_predictions": changed,
    }, indent=2))


if __name__ == "__main__":
    main()
