"""Final held-out sentiment evaluation and paired confidence intervals."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import random
import sys

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path[:0] = [str(RESEARCH / name) for name in (
    "prompt-instruction-ablation", "paired-sentiment-evaluation", "local-sentiment-evaluation"
)]
from evaluate_ablation import build_sample
from compare import align, run_model
from evaluate import LABELS, metrics, sha256


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_intervals(direct, selected, settings):
    pairs = align(direct, selected)
    strata = defaultdict(list)
    for a, b in pairs:
        strata[a["group"], a["label"]].append((a, b))
    rng = random.Random(settings["seed"])
    distributions = {"accuracy": [], "macro_f1": []}
    for _ in range(settings["replicates"]):
        sample = [rng.choice(group) for group in strata.values() for _ in group]
        left = metrics([a for a, _ in sample], "prediction")
        right = metrics([b for _, b in sample], "prediction")
        for name in distributions:
            distributions[name].append(right[name] - left[name])
    tail = (1 - settings["confidence"]) / 2
    left, right = metrics(direct, "prediction"), metrics(selected, "prediction")
    return {
        name: {"selected_minus_direct": right[name] - left[name],
               "interval": [percentile(values, tail), percentile(values, 1 - tail)]}
        for name, values in distributions.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sample-only", action="store_true")
    args = parser.parse_args()
    base_bytes = (RESEARCH / "local-sentiment-evaluation/protocol.json").read_bytes()
    protocol_bytes = (HERE / "protocol.json").read_bytes()
    base, protocol = json.loads(base_bytes), json.loads(protocol_bytes)
    examples, sample = build_sample(args.archive.read_bytes(), base, protocol)
    sample.update({"archive_sha256": base["dataset_sha256"],
                   "base_protocol_sha256": sha256(base_bytes),
                   "protocol_sha256": sha256(protocol_bytes)})
    if args.sample_only:
        print(json.dumps(sample, indent=2))
        return
    if sample != json.loads((HERE / "sample.json").read_bytes()):
        raise ValueError("Sample differs from the committed manifest")

    baseline = {}
    for domain, counts in sample["training_counts"].items():
        label = max(LABELS, key=lambda key: (counts.get(key, 0), key == base["majority_tie_break"]))
        rows = [row | {"prediction": label} for row in examples if row["group"] == domain]
        baseline[domain] = {"label": label, "metrics": metrics(rows, "prediction")}
    identities = {}
    rounds = []
    for order in (protocol["conditions"], list(reversed(protocol["conditions"]))):
        runs = {}
        for model, prompt in order:
            key = model + "/" + prompt
            print(f"Evaluating {key}", file=sys.stderr, flush=True)
            run = run_model(model, examples, base | protocol | {"prompt": protocol["prompts"][prompt]},
                            protocol["warmup"])
            identities.setdefault(model, run["identity"])
            if run["identity"] != identities[model]:
                raise RuntimeError("Model identity changed between conditions")
            run["overall"] = metrics(run["rows"], "prediction")
            runs[key] = run
        rounds.append({"order": order, "runs": runs})
    changed = {
        key: [a["id"] for a, b in align(run["rows"], rounds[1]["runs"][key]["rows"])
              if a["prediction"] != b["prediction"]]
        for key, run in rounds[0]["runs"].items()
    }
    direct = rounds[0]["runs"]["qwen2.5:7b/direct"]["rows"]
    selected = rounds[0]["runs"]["qwen2.5:7b/negation_contrast"]["rows"]
    comparisons = {"overall": paired_intervals(direct, selected, protocol["bootstrap"])}
    for domain in protocol["domains"]:
        comparisons[domain] = paired_intervals(
            [row for row in direct if row["group"] == domain],
            [row for row in selected if row["group"] == domain], protocol["bootstrap"],
        )
    adopt = comparisons["overall"]["accuracy"]["interval"][0] > 0
    print(json.dumps({
        "runtime": {"python": platform.python_version(), "architecture": platform.machine()},
        "sample": sample, "baseline": baseline, "rounds": rounds,
        "changed_predictions": changed,
        "qwen_paired_intervals": comparisons,
        "default_prompts": {"llama3.2:3b": "direct",
                            "qwen2.5:7b": "negation_contrast" if adopt else "direct"},
    }, indent=2))


if __name__ == "__main__":
    main()
