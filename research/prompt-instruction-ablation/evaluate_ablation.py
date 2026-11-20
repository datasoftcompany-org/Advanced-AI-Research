"""Evaluate incremental prompt instructions on unseen review samples."""

import argparse
from collections import Counter
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
from evaluate import LABELS, metrics, read_records, sha256


def text_key(text):
    return sha256(" ".join(text.casefold().split()).encode())


def prior_ids(document):
    if "groups" in document:
        return [row["id"] for row in document["groups"]["test"]["rows"]]
    if "test_ids" in document:
        return document["test_ids"]
    return [row["id"] for row in document["test"]]


def select_sample(records, excluded_hashes, salt, per_label):
    eligible = [row for row in records if text_key(row["text"]) not in excluded_hashes]
    selected, training = [], []
    for label in LABELS:
        rows = sorted(
            (row for row in eligible if row["label"] == label),
            key=lambda row: sha256(f'{salt}:{row["id"]}'.encode()),
        )
        if len(rows) <= per_label:
            raise ValueError(f"Insufficient eligible examples for {label}")
        selected.extend(rows[:per_label])
        training.extend(rows[per_label:])
    return selected, Counter(row["label"] for row in training)


def build_sample(archive, base, protocol):
    sources = {domain: read_records(archive, base | settings)
               for domain, settings in protocol["domains"].items()}
    lookup = {row["id"]: row for rows in sources.values() for row in rows}
    exclusions = set()
    provenance = {}
    for relative in protocol["previous_samples"]:
        raw = (RESEARCH / relative).read_bytes()
        provenance[relative] = sha256(raw)
        exclusions.update(text_key(lookup[key]["text"]) for key in prior_ids(json.loads(raw)))

    examples = []
    counts = {}
    # Also prevent normalized duplicates across the new domain samples.
    used = set(exclusions)
    for domain, rows in sources.items():
        selected, counts[domain] = select_sample(
            rows, used, protocol["split_salt"], protocol["test_per_label"]
        )
        examples.extend(row | {"group": domain} for row in selected)
        used.update(text_key(row["text"]) for row in selected)
    return examples, {
        "previous_sample_hashes": provenance,
        "excluded_text_hashes": sorted(exclusions),
        "training_counts": counts,
        "test": [{"id": row["id"], "group": row["group"], "label": row["label"],
                  "text_sha256": text_key(row["text"])} for row in examples],
    }


def wins(left, right):
    return {
        "left_only": [a["id"] for a, b in align(left, right)
                      if a["prediction"] == a["label"] and b["prediction"] != b["label"]],
        "right_only": [a["id"] for a, b in align(left, right)
                       if b["prediction"] == b["label"] and a["prediction"] != a["label"]],
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
    conditions = [(model, prompt) for model in protocol["models"] for prompt in protocol["prompts"]]
    identities = {}
    rounds = []
    for order in (conditions, list(reversed(conditions))):
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
        comparisons = {}
        names = list(protocol["prompts"])
        for model in protocol["models"]:
            comparisons[model] = {
                left + "_to_" + right: wins(runs[model + "/" + left]["rows"],
                                           runs[model + "/" + right]["rows"])
                for left, right in zip(names, names[1:])
            }
        rounds.append({"order": order, "runs": runs, "adjacent_comparisons": comparisons})
    changed = {
        key: [a["id"] for a, b in align(run["rows"], rounds[1]["runs"][key]["rows"])
              if a["prediction"] != b["prediction"]]
        for key, run in rounds[0]["runs"].items()
    }
    print(json.dumps({
        "runtime": {"python": platform.python_version(), "architecture": platform.machine()},
        "sample": sample, "baseline": baseline, "rounds": rounds,
        "changed_predictions": changed,
    }, indent=2))


if __name__ == "__main__":
    main()
