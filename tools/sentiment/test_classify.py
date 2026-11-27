"""Regression coverage for CSV handling and local service failures."""

import argparse
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import classify


class ToolTests(unittest.TestCase):
    def test_csv_preserves_quotes_multiline_and_empty_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reviews.csv"
            source.write_text('\ufefftext\n"Good, indeed\nstill good"\n""\n', encoding="utf-8")
            args = argparse.Namespace(text=None, csv=source, column="text")
            self.assertEqual(classify.load_texts(args), ["Good, indeed\nstill good", ""])

    def test_csv_missing_column_and_extra_cells_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reviews.csv"
            args = argparse.Namespace(text=None, csv=source, column="text")
            for content in ("review\ngood\n", "text\ngood,extra\n"):
                source.write_text(content)
                with self.assertRaises(ValueError):
                    classify.load_texts(args)

    def test_failure_and_invalid_responses_remain_visible(self):
        identity = {"name": "local", "digest": "hash", "ollama": "version"}
        with patch.object(classify, "model_identity", return_value=identity), patch.object(
            classify, "predict", side_effect=["positive", "failure", "invalid"]
        ) as predict:
            rows = classify.classify(["good", "bad", "unknown", ""], {})
        self.assertEqual([row["status"] for row in rows], ["ok", "failure", "invalid", "empty"])
        self.assertEqual(predict.call_count, 3)
        self.assertEqual(rows[1]["sentiment"], "")

    def test_identity_changes_abort_the_batch(self):
        before = {"name": "local", "digest": "first", "ollama": "version"}
        with patch.object(classify, "model_identity", side_effect=[before, before | {"digest": "new"}]), patch.object(
            classify, "predict", return_value="positive"
        ), self.assertRaisesRegex(ValueError, "changed"):
            classify.classify(["good"], {})

    def test_export_is_complete_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.csv"
            classify.export_csv([{"row": 1, "text": "=1+1", "status": "empty"}], output)
            original = output.read_bytes()
            with output.open(newline="") as stream:
                self.assertEqual(next(csv.DictReader(stream))["text"], "'=1+1")
            with self.assertRaises(FileExistsError):
                classify.export_csv([], output)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(output.parent.iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
