"""Compare quality, latency and Ollama-reported allocation on fixed texts."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "research" / name) for name in (
    "paired-sentiment-evaluation", "local-sentiment-evaluation"
)]
from compare import align
from evaluate import LABELS, metrics, model_identity, predict, request, sha256


def percentile95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def allocation(models, name):
    matches = [model for model in models if model.get("name") == name or model.get("model") == name]
    if len(matches) != 1:
        raise ValueError("Loaded target model allocation unavailable")
    record = matches[0]
    return {key: record.get(key) for key in ("size", "size_vram", "context_length")}


def examples_from(raw):
    texts = list(csv.DictReader(io.StringIO(raw["reviews"].decode(), newline="")))
    references = list(csv.DictReader(io.StringIO(raw["references"].decode(), newline="")))
    if len(texts) != len(references):
        raise ValueError("Acceptance source counts differ")
    examples = []
    for index, (text, reference) in enumerate(zip(texts, references), 1):
        if reference["row"] != str(index):
            raise ValueError("Reference row order differs")
        if reference["status"] == "ok":
            examples.append({"id": "acceptance:" + str(index), "text": text["text"],
                             "label": reference["sentiment"], "group": "acceptance"})
    for case in json.loads(raw["stability"]):
        examples.append({"id": case["id"] + "/typo", "text": case["variants"]["typo"],
                         "label": case["label"], "group": "spelling"})
    if any(row["label"] not in LABELS for row in examples):
        raise ValueError("Invalid reference label")
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    paths = {
        "base": ROOT / "research/local-sentiment-evaluation/protocol.json",
        "protocol": HERE / "protocol.json",
        "reviews": ROOT / "tools/sentiment/acceptance/reviews.csv",
        "references": ROOT / "tools/sentiment/acceptance/expected.csv",
        "stability": ROOT / "research/sentiment-stability/cases.json",
    }
    raw = {key: path.read_bytes() for key, path in paths.items()}
    manifest = {key: sha256(value) for key, value in raw.items()}
    if args.manifest_only:
        print(json.dumps(manifest, indent=2))
        return
    if manifest != json.loads((HERE / "manifest.json").read_bytes()):
        raise ValueError("Benchmark sources differ from frozen manifest")
    base, config = json.loads(raw["base"]), json.loads(raw["protocol"])
    examples = examples_from(raw)
    if request("/api/ps", 5).get("models"):
        raise ValueError("Begin with no loaded Ollama models to measure load and allocation.")
    try:
        ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        ram = None
    identities = {}
    rounds = []
    conditions = [(model, context) for model in config["models"] for context in config["contexts"]]
    for order in (conditions, list(reversed(conditions))):
        runs = {}
        for model, context in order:
            key = model + "/" + str(context)
            print(f"Benchmarking {key}", file=sys.stderr, flush=True)
            settings = base | {"model": model, "options": base["options"] | {"num_ctx": context}}
            identity = model_identity(settings)
            identities.setdefault(model, identity)
            if identities[model] != identity:
                raise ValueError("Model identity changed")
            rows, snapshots = [], []
            try:
                start = time.perf_counter()
                if predict(config["warmup"], settings) not in LABELS:
                    raise ValueError("Warm-up failed")
                cold_seconds = time.perf_counter() - start
                snapshots.append(allocation(request("/api/ps", 5)["models"], model))
                for example in examples:
                    start = time.perf_counter()
                    answer = predict(example["text"], settings)
                    elapsed = time.perf_counter() - start
                    rows.append({key: example[key] for key in ("id", "group", "label")} |
                                {"prediction": answer, "seconds": round(elapsed, 6)})
                    snapshots.append(allocation(request("/api/ps", 5)["models"], model))
                if model_identity(settings) != identity:
                    raise ValueError("Model identity changed during run")
            finally:
                request("/api/generate", 30, {"model": model, "keep_alive": 0, "stream": False})
            elapsed = [row["seconds"] for row in rows if row["prediction"] != "failure"]
            runs[key] = {
                "identity": identity, "context": context,
                "load_and_warmup_seconds": cold_seconds,
                "quality": metrics(rows, "prediction"),
                "group_quality": {group: metrics([row for row in rows if row["group"] == group], "prediction")
                                  for group in sorted({row["group"] for row in rows})},
                "median_seconds": statistics.median(elapsed) if elapsed else None,
                "p95_seconds": percentile95(elapsed) if elapsed else None,
                "allocation_snapshots": snapshots,
                "max_reported_bytes": {
                    field: max(values) if values else None
                    for field in ("size", "size_vram")
                    for values in [[s[field] for s in snapshots if isinstance(s[field], int)]]
                },
                "rows": rows,
            }
        rounds.append({"order": order, "runs": runs})
    changed = {
        key: [a["id"] for a, b in align(run["rows"], rounds[1]["runs"][key]["rows"])
              if a["prediction"] != b["prediction"]]
        for key, run in rounds[0]["runs"].items()
    }
    print(json.dumps({
        "runtime": {"python": platform.python_version(), "architecture": platform.machine(),
                    "system": platform.platform(), "physical_ram_bytes": ram},
        "manifest": manifest, "rounds": rounds, "changed_predictions": changed,
    }, indent=2))


if __name__ == "__main__":
    main()
