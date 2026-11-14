"""Evaluate a local Ollama model on a fixed English sentiment split."""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import platform
import urllib.error
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
LABELS = ("negative", "positive")
ENDPOINT = "http://127.0.0.1:11434"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_split(archive, protocol):
    if sha256(archive) != protocol["dataset_sha256"]:
        raise ValueError("Dataset archive SHA-256 does not match the protocol")
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        lines = source.read(protocol["member"]).decode("utf-8").splitlines()

    groups = defaultdict(list)
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        text, label = line.rsplit("\t", 1)
        normalized = " ".join(text.casefold().split())
        groups[normalized].append({
            "id": f"yelp:{index}", "text": text.strip(), "label": {"0": "negative", "1": "positive"}[label],
        })

    test, training = [], []
    for label in LABELS:
        rows = [
            rows[0] for rows in groups.values()
            if {row["label"] for row in rows} == {label}
        ]
        def rank(row):
            return sha256(f'{protocol["split_salt"]}:{row["id"]}'.encode())
        rows.sort(key=rank)
        cutoff = protocol["test_per_label"]
        if len(rows) <= cutoff:
            raise ValueError(f"Not enough distinct {label} rows")
        test.extend(rows[:cutoff])
        training.extend(rows[cutoff:])
    return test, Counter(row["label"] for row in training)


def metrics(rows, field):
    matrix = {label: dict.fromkeys((*LABELS, "invalid", "failure"), 0) for label in LABELS}
    for row in rows:
        matrix[row["label"]][row[field]] += 1
    correct = sum(matrix[label][label] for label in LABELS)
    f1 = {}
    for label in LABELS:
        denominator = sum(matrix[label].values()) + sum(matrix[actual][label] for actual in LABELS)
        f1[label] = 2 * matrix[label][label] / denominator if denominator else 0
    return {
        "n": len(rows),
        "accuracy": correct / len(rows),
        "macro_f1": sum(f1.values()) / len(LABELS),
        "invalid": sum(matrix[label]["invalid"] for label in LABELS),
        "failures": sum(matrix[label]["failure"] for label in LABELS),
        "confusion": matrix,
    }


def request(path, timeout, body=None):
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(ENDPOINT + path, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("Ollama response must be a JSON object")
    return result


def predict(text, protocol):
    try:
        result = request("/api/generate", protocol["timeout_seconds"], {
            "model": protocol["model"],
            "prompt": protocol["prompt"] + json.dumps(text, ensure_ascii=False),
            "options": protocol["options"],
            "stream": False,
            "keep_alive": protocol["keep_alive"],
        })
        if result.get("done") is not True or not isinstance(result.get("response"), str):
            return "failure"
    except (OSError, ValueError, urllib.error.URLError):
        return "failure"
    answer = result["response"].strip()
    return answer if answer in LABELS else "invalid"


def model_identity(protocol):
    version = request("/api/version", 5)["version"]
    for model in request("/api/tags", 5)["models"]:
        if model["name"] == protocol["model"] and not model.get("remote_host"):
            return {"name": model["name"], "digest": model["digest"], "ollama": version}
    raise ValueError(f'{protocol["model"]} must already be installed locally')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Official UCI dataset ZIP")
    args = parser.parse_args()
    protocol_bytes = (HERE / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    diagnostics_bytes = (HERE / "diagnostics.json").read_bytes()
    test, counts = read_split(args.archive.read_bytes(), protocol)
    majority = max(LABELS, key=lambda label: (counts[label], label == protocol["majority_tie_break"]))
    identity = model_identity(protocol)
    groups = {"test": test}
    for row in json.loads(diagnostics_bytes):
        groups.setdefault(row["group"], []).append(row)

    reports = {}
    for name, group in groups.items():
        rows = [{
            "id": row["id"], "label": row["label"], "baseline": majority,
            "model": predict(row["text"], protocol),
        } for row in group]
        reports[name] = {**{key: metrics(rows, key) for key in ("baseline", "model")}, "rows": rows}

    if model_identity(protocol) != identity:
        raise ValueError("Ollama version or model digest changed during evaluation")
    print(json.dumps({
        "runtime": {"python": platform.python_version(), "architecture": platform.machine()},
        "model": identity,
        "archive_sha256": protocol["dataset_sha256"],
        "protocol_sha256": sha256(protocol_bytes),
        "diagnostics_sha256": sha256(diagnostics_bytes),
        "training_counts": counts,
        "majority_class": majority,
        "groups": reports,
    }, indent=2))


if __name__ == "__main__":
    main()
