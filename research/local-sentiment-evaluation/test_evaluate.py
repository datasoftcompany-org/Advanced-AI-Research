"""Regression checks for scoring and the local response boundary."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import evaluate


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(Path(evaluate.HERE, "protocol.json").read_text())

    def test_invalid_and_failed_predictions_count_as_errors(self):
        rows = [
            {"label": "negative", "model": "negative"},
            {"label": "negative", "model": "invalid"},
            {"label": "positive", "model": "positive"},
            {"label": "positive", "model": "failure"},
        ]
        scores = evaluate.metrics(rows, "model")
        self.assertEqual(scores["accuracy"], 0.5)
        self.assertAlmostEqual(scores["macro_f1"], 2 / 3)
        self.assertEqual(scores["invalid"], 1)
        self.assertEqual(scores["failures"], 1)

    def test_prediction_only_accepts_label_after_trimming_whitespace(self):
        with patch.object(evaluate, "request", return_value={"done": True, "response": " positive\n"}):
            self.assertEqual(evaluate.predict("Review", self.protocol), "positive")
        with patch.object(evaluate, "request", return_value={"done": True, "response": "positive."}):
            self.assertEqual(evaluate.predict("Review", self.protocol), "invalid")

    def test_interrupted_service_is_a_failure(self):
        with patch.object(evaluate, "request", side_effect=TimeoutError):
            self.assertEqual(evaluate.predict("Review", self.protocol), "failure")
        with patch.object(evaluate, "request", return_value={"done": False, "response": "positive"}):
            self.assertEqual(evaluate.predict("Review", self.protocol), "failure")

    def test_archive_checksum_mismatch_fails_before_extraction(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            evaluate.read_split(b"wrong archive", self.protocol)


if __name__ == "__main__":
    unittest.main()
