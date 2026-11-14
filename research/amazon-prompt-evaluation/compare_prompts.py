"""Evaluate two prompt formulations on a fixed Amazon review sample."""

import argparse
import json
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path[:0] = [str(RESEARCH / name) for name in (
    "paired-sentiment-evaluation", "local-sentiment-evaluation"
)]
from compare import align, run_model
from evaluate import LABELS, metrics, read_split, sha256


def paired_outcomes(direct, overall):
    pairs = align(direct, overall)
    return {
        "direct_only": [a["id"] for a, b in pairs
                        if a["prediction"] == a["label"] and b["prediction"] != b["label"]],
        "overall_only": [a["id"] for a, b in pairs
                         if b["prediction"] == b["label"] and a["prediction"] != a["label"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sample-only", action="store_true")
    args = parser.parse_args()
    inherited_bytes = (RESEARCH / "local-sentiment-evaluation/protocol.json").read_bytes()
    protocol_bytes = (HERE / "protocol.json").read_bytes()
    inherited = json.loads(inherited_bytes)
    protocol = json.loads(protocol_bytes)
    settings = inherited | protocol
    test, counts = read_split(args.archive.read_bytes(), settings)
    examples = [row | {"group": "test"} for row in test]
    sample = {
        "archive_sha256": settings["dataset_sha256"],
        "inherited_protocol_sha256": sha256(inherited_bytes),
        "protocol_sha256": sha256(protocol_bytes),
        "training_counts": counts,
        "test_ids": [row["id"] for row in test],
    }
    if args.sample_only:
        print(json.dumps(sample, indent=2))
        return
    if sample != json.loads((HERE / "sample.json").read_bytes()):
        raise ValueError("Sample or protocol differs from the frozen manifest")

    baseline = max(LABELS, key=lambda label: (counts[label], label == settings["majority_tie_break"]))
    prompts = {"direct": inherited["prompt"], "overall": protocol["overall_prompt"]}
    rounds = []
    identity = None
    for order in protocol["orders"]:
        runs = {}
        for name in order:
            print(f"Evaluating {name}", file=sys.stderr, flush=True)
            run = run_model(settings["model"], examples, settings | {"prompt": prompts[name]}, protocol["warmup"])
            identity = identity or run["identity"]
            if run["identity"] != identity:
                raise RuntimeError("Model identity changed between runs")
            runs[name] = run
        rounds.append({"order": order, "runs": runs,
                       "paired_outcomes": paired_outcomes(runs["direct"]["rows"], runs["overall"]["rows"])})
    changed = {}
    for name in prompts:
        first, second = (round_["runs"][name]["rows"] for round_ in rounds)
        changed[name] = [a["id"] for a, b in align(first, second) if a["prediction"] != b["prediction"]]
    print(json.dumps({
        "runtime": {"python": platform.python_version(), "architecture": platform.machine()},
        "sample": sample,
        "baseline_label": baseline,
        "baseline": metrics([row | {"prediction": baseline} for row in test], "prediction"),
        "rounds": rounds,
        "changed_predictions": changed,
    }, indent=2))


if __name__ == "__main__":
    main()
