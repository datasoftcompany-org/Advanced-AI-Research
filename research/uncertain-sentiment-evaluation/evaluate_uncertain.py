"""Compare forced binary decisions with explicit human-review deferral."""

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / name) for name in (
    "paired-sentiment-evaluation", "local-sentiment-evaluation"
)]
from compare import align
from evaluate import LABELS, model_identity, predict, request, sha256


def summarize(rows):
    clear = [row for row in rows if row["reference"] in LABELS]
    ambiguous = [row for row in rows if row["reference"] == "uncertain"]
    counts = Counter(row["prediction"] for row in rows)
    correct_clear = sum(row["prediction"] == row["reference"] for row in clear)
    referred = sum(row["prediction"] == "uncertain" for row in ambiguous)
    return {
        "n": len(rows), "counts": dict(counts),
        "clear_correct": correct_clear, "clear_total": len(clear),
        "clear_accuracy": correct_clear / len(clear),
        "ambiguous_referred": referred, "ambiguous_total": len(ambiguous),
        "ambiguous_referral_rate": referred / len(ambiguous),
        "clear_false_referrals": sum(row["prediction"] == "uncertain" for row in clear),
        "ambiguous_forced_polar": sum(row["prediction"] in LABELS for row in ambiguous),
        "polar_coverage": sum(counts[label] for label in LABELS) / len(rows),
        "median_seconds": statistics.median(row["seconds"] for row in rows),
        "categories": {
            category: {"correct": sum(row["prediction"] == row["reference"]
                                     for row in rows if row["category"] == category),
                       "total": sum(row["category"] == category for row in rows)}
            for category in sorted({row["category"] for row in rows})
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    files = {"base": HERE.parent / "local-sentiment-evaluation/protocol.json",
             "protocol": HERE / "protocol.json", "cases": HERE / "cases.json"}
    raw = {key: path.read_bytes() for key, path in files.items()}
    manifest = {key: sha256(value) for key, value in raw.items()}
    if args.manifest_only:
        print(json.dumps(manifest, indent=2))
        return
    if manifest != json.loads((HERE / "manifest.json").read_bytes()):
        raise ValueError("Cases or protocol differ from frozen manifest")
    base, config, cases = (json.loads(raw[key]) for key in ("base", "protocol", "cases"))
    identities = {}
    rounds = []
    conditions = [(model, mode) for model in config["models"] for mode in config["prompts"]]
    for order in (conditions, list(reversed(conditions))):
        runs = {}
        for model, mode in order:
            key = model + "/" + mode
            print(f"Evaluating {key}", file=sys.stderr, flush=True)
            settings = base | {"model": model, "prompt": config["prompts"][mode]}
            identity = model_identity(settings)
            identities.setdefault(model, identity)
            if identities[model] != identity:
                raise ValueError("Model identity changed")
            allowed = LABELS if mode == "binary" else (*LABELS, "uncertain")
            rows = []
            try:
                if predict(config["warmup"], settings, allowed) not in LABELS:
                    raise ValueError("Warmup failed")
                for case in cases:
                    started = time.perf_counter()
                    answer = predict(case["text"], settings, allowed)
                    rows.append({"id": case["id"], "group": case["category"],
                                 "category": case["category"], "label": case["reference"],
                                 "reference": case["reference"], "prediction": answer,
                                 "seconds": round(time.perf_counter() - started, 6)})
                if model_identity(settings) != identity:
                    raise ValueError("Model identity changed during run")
            finally:
                request("/api/generate", 30, {"model": model, "keep_alive": 0, "stream": False})
            runs[key] = {"identity": identity, "summary": summarize(rows), "rows": rows}
        rounds.append({"order": order, "runs": runs})
    changed = {
        key: [a["id"] for a, b in align(run["rows"], rounds[1]["runs"][key]["rows"])
              if a["prediction"] != b["prediction"]]
        for key, run in rounds[0]["runs"].items()
    }
    print(json.dumps({"runtime": {"python": platform.python_version(), "architecture": platform.machine()},
                      "manifest": manifest, "rounds": rounds, "changed_predictions": changed}, indent=2))


if __name__ == "__main__":
    main()
