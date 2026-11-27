"""Measure sentiment consistency under authored meaning-preserving edits."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / name) for name in (
    "paired-sentiment-evaluation", "local-sentiment-evaluation"
)]
from compare import align, run_model
from evaluate import LABELS, sha256


def examples_from(cases):
    examples = []
    seen = set()
    for case in cases:
        if case["id"] in seen or case["label"] not in LABELS:
            raise ValueError("Duplicate case ID or invalid reference")
        seen.add(case["id"])
        if set(case["variants"]) != {"paraphrase", "typo", "punctuation"}:
            raise ValueError("Missing transformation")
        for kind, text in {"original": case["original"], **case["variants"]}.items():
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty example")
            examples.append({"id": case["id"] + "/" + kind, "text": text,
                             "label": case["label"], "group": kind})
    return examples


def paired_summary(cases, rows):
    lookup = {row["id"]: row for row in rows}
    expected = examples_from(cases)
    if len(lookup) != len(rows) or set(lookup) != {row["id"] for row in expected}:
        raise ValueError("Predictions do not match the fixture")
    if any(lookup[row["id"]]["label"] != row["label"] for row in expected):
        raise ValueError("Prediction reference differs from fixture")
    groups = defaultdict(list)
    for case in cases:
        original = lookup[case["id"] + "/original"]["prediction"]
        for kind in case["variants"]:
            variant = lookup[case["id"] + "/" + kind]["prediction"]
            groups[kind].append({
                "id": case["id"], "label": case["label"],
                "original": original, "variant": variant,
                "valid_agreement": original in LABELS and variant in LABELS and original == variant,
                "both_correct": original == variant == case["label"],
                "correct_to_wrong": original == case["label"] and variant != case["label"],
                "wrong_to_correct": original != case["label"] and variant == case["label"],
                "either_unusable": original not in LABELS or variant not in LABELS,
            })
    return {
        kind: {
            "n": len(pairs),
            **{key: sum(pair[key] for pair in pairs)
               for key in ("valid_agreement", "both_correct", "correct_to_wrong",
                           "wrong_to_correct", "either_unusable")},
            "agreement_rate": sum(pair["valid_agreement"] for pair in pairs) / len(pairs),
            "pairs": pairs,
        } for kind, pairs in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    files = {"base": HERE.parent / "local-sentiment-evaluation/protocol.json",
             "cases": HERE / "cases.json", "protocol": HERE / "protocol.json"}
    raw = {key: path.read_bytes() for key, path in files.items()}
    manifest = {key: sha256(value) for key, value in raw.items()}
    if args.manifest_only:
        print(json.dumps(manifest, indent=2))
        return
    if manifest != json.loads((HERE / "manifest.json").read_bytes()):
        raise ValueError("Inputs differ from frozen manifest")
    base, cases, protocol = (json.loads(raw[key]) for key in ("base", "cases", "protocol"))
    examples = examples_from(cases)
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
            run["stability"] = paired_summary(cases, run["rows"])
            runs[model] = run
        rounds.append({"order": order, "runs": runs})
    changed = {
        model: [a["id"] for a, b in align(run["rows"], rounds[1]["runs"][model]["rows"])
                if a["prediction"] != b["prediction"]]
        for model, run in rounds[0]["runs"].items()
    }
    print(json.dumps({"runtime": {"python": platform.python_version(), "architecture": platform.machine()},
                      "manifest": manifest, "rounds": rounds, "changed_predictions": changed}, indent=2))


if __name__ == "__main__":
    main()
