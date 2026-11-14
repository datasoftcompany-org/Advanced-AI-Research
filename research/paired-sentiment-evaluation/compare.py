"""Compare fixed local models in two reversed, sequential rounds."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import platform
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "local-sentiment-evaluation"
sys.path.insert(0, str(PREVIOUS))
from evaluate import LABELS, metrics, model_identity, predict, read_split, request, sha256


def align(left, right):
    lookup = {row["id"]: row for row in right}
    if len(lookup) != len(right) or len({row["id"] for row in left}) != len(left):
        raise ValueError("Duplicate example IDs")
    if {row["id"] for row in left} != lookup.keys():
        raise ValueError("Prediction sets contain different example IDs")
    pairs = [(row, lookup[row["id"]]) for row in left]
    if any(a["label"] != b["label"] or a["group"] != b["group"] for a, b in pairs):
        raise ValueError("Reference labels or groups differ")
    return pairs


def load_sample(archive, previous, protocol):
    prior, _ = read_split(archive, previous)
    excluded = {row["id"] for row in prior}
    test, training = read_split(archive, previous | protocol, excluded)
    return test, training, sorted(excluded)


def run_model(model, examples, settings, warmup):
    settings = settings | {"model": model}
    identity = model_identity(settings)
    try:
        if predict(warmup, settings) not in LABELS:
            raise RuntimeError(f"{model}: warm-up did not return a valid label")
        rows = []
        for row in examples:
            started = time.perf_counter()
            prediction = predict(row["text"], settings)
            elapsed = time.perf_counter() - started
            rows.append({
                "id": row["id"], "group": row["group"], "label": row["label"],
                "prediction": prediction, "seconds": round(elapsed, 6),
            })
        if model_identity(settings) != identity:
            raise RuntimeError(f"{model}: identity changed during evaluation")
    finally:
        request("/api/generate", 30, {"model": model, "keep_alive": 0, "stream": False})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    summary = {}
    for group, members in grouped.items():
        latencies = [row["seconds"] for row in members if row["prediction"] != "failure"]
        summary[group] = metrics(members, "prediction") | {
            "median_seconds": statistics.median(latencies) if latencies else None,
            "timed_responses": len(latencies),
        }
    return {"identity": identity, "metrics": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sample-only", action="store_true", help="Print the frozen sample manifest")
    args = parser.parse_args()
    previous_bytes = (PREVIOUS / "protocol.json").read_bytes()
    protocol_bytes = (HERE / "protocol.json").read_bytes()
    diagnostic_bytes = (HERE / "diagnostics.tsv").read_bytes()
    previous = json.loads(previous_bytes)
    protocol = json.loads(protocol_bytes)
    test, training, excluded = load_sample(args.archive.read_bytes(), previous, protocol)
    diagnostic = list(csv.DictReader(diagnostic_bytes.decode().splitlines(), delimiter="\t"))
    groups = {"test": [row | {"group": "test"} for row in test]}
    for row in diagnostic:
        groups.setdefault(row["group"], []).append(row)
    manifest = {
        "dataset_sha256": previous["dataset_sha256"],
        "previous_protocol_sha256": sha256(previous_bytes),
        "protocol_sha256": sha256(protocol_bytes),
        "diagnostics_sha256": sha256(diagnostic_bytes),
        "excluded_ids": excluded,
        "test_ids": [row["id"] for row in test],
        "training_counts": training,
    }
    if args.sample_only:
        print(json.dumps(manifest, indent=2))
        return
    if manifest != json.loads((HERE / "sample.json").read_bytes()):
        raise ValueError("Data or protocol differs from the frozen sample manifest")
    baseline = max(LABELS, key=lambda label: (training[label], label == previous["majority_tie_break"]))
    examples = [row for group in groups.values() for row in group]
    rounds = []
    for order in protocol["orders"]:
        models = {}
        for model in order:
            print(f"Evaluating {model}", file=sys.stderr, flush=True)
            models[model] = run_model(model, examples, previous | protocol, protocol["warmup"])
        llama, qwen = (models[m]["rows"] for m in protocol["orders"][0])
        disagreements = [
            {"id": a["id"], "group": a["group"], "label": a["label"],
             "llama": a["prediction"], "qwen": b["prediction"]}
            for a, b in align(llama, qwen) if a["prediction"] != b["prediction"]
        ]
        rounds.append({"order": order, "models": models, "disagreements": disagreements})
    changed = {}
    for model in protocol["orders"][0]:
        first, second = (r["models"][model] for r in rounds)
        if first["identity"] != second["identity"]:
            raise RuntimeError(f"{model}: identity changed between rounds")
        changed[model] = [a["id"] for a, b in align(first["rows"], second["rows"])
                          if a["prediction"] != b["prediction"]]
    print(json.dumps({
        "runtime": {"python": platform.python_version(), "architecture": platform.machine()},
        "sample": manifest,
        "baseline_label": baseline,
        "baseline": {name: metrics([row | {"prediction": baseline} for row in rows], "prediction")
                     for name, rows in groups.items()},
        "rounds": rounds,
        "changed_predictions": changed,
    }, indent=2))


if __name__ == "__main__":
    main()
