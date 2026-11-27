"""Classify short English reviews locally using the validated direct prompt."""

import argparse
import csv
import json
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/local-sentiment-evaluation"))
from evaluate import model_identity, predict

MODELS = ("llama3.2:3b", "qwen2.5:7b")
FIELDS = ("row", "text", "sentiment", "status", "model", "model_digest", "ollama_version")


def load_texts(args):
    if args.text is not None:
        if not args.text.strip():
            raise ValueError("Enter a non-empty English review.")
        return [args.text]
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        return read_csv(stream, args.column)


def read_csv(stream, column):
    reader = csv.DictReader(stream)
    if not reader.fieldnames or column not in reader.fieldnames:
        raise ValueError(f"CSV must contain column {column!r}.")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("CSV contains duplicate column names.")
    texts = []
    for row in reader:
        if None in row or row[column] is None:
            raise ValueError("CSV contains a row with missing or extra cells.")
        texts.append(row[column])
    if not texts:
        raise ValueError("CSV contains no data rows.")
    return texts


def classify(texts, settings, progress=None):
    identity = model_identity(settings)
    rows = []
    for index, text in enumerate(texts, 1):
        answer = predict(text, settings) if text.strip() else "empty"
        valid = answer in ("positive", "negative")
        rows.append({
            "row": index, "text": text, "sentiment": answer if valid else "",
            "status": "ok" if valid else answer, "model": identity["name"],
            "model_digest": identity["digest"], "ollama_version": identity["ollama"],
        })
        if progress:
            progress(index, len(texts))
    if model_identity(settings) != identity:
        raise ValueError("Model or Ollama version changed during classification. Retry the batch.")
    return rows


def spreadsheet_safe(value):
    text = str(value)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def csv_text(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows({key: spreadsheet_safe(value) for key, value in row.items()} for row in rows)
    return buffer.getvalue()


def export_csv(rows, output):
    """Write a complete export without replacing an existing destination."""
    if not output.parent.is_dir():
        raise ValueError("Output directory does not exist.")
    content = csv_text(rows)
    # Exclusive creation also works on the shared exFAT archive disk.
    with output.open("x", encoding="utf-8", newline="") as stream:
        try:
            stream.write(content)
            stream.flush()
        except OSError:
            output.unlink()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="One short English review")
    source.add_argument("--csv", type=Path, help="UTF-8 CSV input")
    parser.add_argument("--column", default="text", help="Review column (default: text)")
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--output", type=Path, help="New results CSV; existing files are preserved")
    args = parser.parse_args()
    try:
        texts = load_texts(args)
        if args.output and args.output.exists():
            raise ValueError("Output already exists. Choose a new filename.")
        settings = json.loads((ROOT / "research/local-sentiment-evaluation/protocol.json").read_text())
        settings["model"] = args.model
        rows = classify(texts, settings)
        if args.output:
            export_csv(rows, args.output)
            print(f"Saved {len(rows)} rows to {args.output}", file=sys.stderr)
        else:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 2 if any(row["status"] != "ok" for row in rows) else 0
    except (OSError, ValueError, KeyError, csv.Error) as error:
        print(f"Classification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
