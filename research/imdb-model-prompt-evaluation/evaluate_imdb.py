"""Compare two local models and two prompts on a fixed IMDb sample."""

import argparse
import json
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path[:0] = [str(RESEARCH / name) for name in (
    "amazon-prompt-evaluation", "paired-sentiment-evaluation", "local-sentiment-evaluation"
)]
from compare import align, run_model
from compare_prompts import paired_outcomes
from evaluate import LABELS, metrics, read_split, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sample-only", action="store_true")
    args = parser.parse_args()
    files = {
        "base": RESEARCH / "local-sentiment-evaluation/protocol.json",
        "amazon": RESEARCH / "amazon-prompt-evaluation/protocol.json",
        "imdb": HERE / "protocol.json",
    }
    raw = {key: path.read_bytes() for key, path in files.items()}
    protocols = {key: json.loads(value) for key, value in raw.items()}
    settings = protocols["base"] | protocols["imdb"]
    prompts = {
        "direct": protocols["base"]["prompt"],
        "overall": protocols["amazon"]["overall_prompt"],
    }
    test, counts = read_split(args.archive.read_bytes(), settings)
    examples = [row | {"group": "test"} for row in test]
    sample = {
        "archive_sha256": settings["dataset_sha256"],
        "protocol_hashes": {key: sha256(value) for key, value in raw.items()},
        "training_counts": counts,
        "test": [{"id": row["id"], "label": row["label"],
                  "text_sha256": sha256(row["text"].encode())} for row in test],
    }
    if args.sample_only:
        print(json.dumps(sample, indent=2))
        return
    if sample != json.loads((HERE / "sample.json").read_bytes()):
        raise ValueError("Sample differs from the committed manifest")

    baseline = max(LABELS, key=lambda label: (counts[label], label == settings["majority_tie_break"]))
    identities = {}
    rounds = []
    for order in settings["orders"]:
        runs = {}
        for model, prompt in order:
            key = model + "/" + prompt
            print(f"Evaluating {key}", file=sys.stderr, flush=True)
            run = run_model(model, examples, settings | {"prompt": prompts[prompt]}, settings["warmup"])
            identities.setdefault(model, run["identity"])
            if identities[model] != run["identity"]:
                raise RuntimeError("Model identity changed between conditions")
            runs[key] = run
        prompt_comparisons = {
            model: paired_outcomes(runs[model + "/direct"]["rows"], runs[model + "/overall"]["rows"])
            for model in identities
        }
        model_comparisons = {
            prompt: {
                "llama_only": [a["id"] for a, b in pairs
                               if a["prediction"] == a["label"] and b["prediction"] != b["label"]],
                "qwen_only": [a["id"] for a, b in pairs
                              if b["prediction"] == b["label"] and a["prediction"] != a["label"]],
            }
            for prompt in prompts
            for pairs in [align(runs["llama3.2:3b/" + prompt]["rows"],
                                runs["qwen2.5:7b/" + prompt]["rows"])]
        }
        rounds.append({"order": order, "runs": runs, "prompt_comparisons": prompt_comparisons,
                       "model_comparisons": model_comparisons})
    changed = {
        key: [a["id"] for a, b in align(run["rows"], rounds[1]["runs"][key]["rows"])
              if a["prediction"] != b["prediction"]]
        for key, run in rounds[0]["runs"].items()
    }
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
