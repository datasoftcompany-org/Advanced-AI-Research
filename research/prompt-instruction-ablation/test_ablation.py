"""Guard text-level exclusion and correctness comparisons."""

import unittest

from evaluate_ablation import prior_ids, select_sample, text_key, wins


class AblationTests(unittest.TestCase):
    def test_excludes_same_text_under_a_different_id(self):
        records = [
            {"id": "new:1", "text": " Good   MEAL ", "label": "positive"},
            {"id": "new:2", "text": "Fine", "label": "positive"},
            {"id": "new:3", "text": "Excellent", "label": "positive"},
            {"id": "new:4", "text": "Bad", "label": "negative"},
            {"id": "new:5", "text": "Poor", "label": "negative"},
        ]
        selected, counts = select_sample(records, {text_key("good meal")}, "fixed", 1)
        self.assertNotIn("new:1", [row["id"] for row in selected])
        self.assertEqual(len(selected), 2)
        self.assertEqual(dict(counts), {"negative": 1, "positive": 1})

    def test_sample_shortage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            select_sample([], set(), "fixed", 1)

    def test_reads_all_historical_manifest_formats(self):
        self.assertEqual(prior_ids({"test_ids": ["a"]}), ["a"])
        self.assertEqual(prior_ids({"test": [{"id": "b"}]}), ["b"])
        self.assertEqual(prior_ids({"groups": {"test": {"rows": [{"id": "c"}]}}}), ["c"])

    def test_wins_are_matched_by_id_and_invalid_counts_as_wrong(self):
        a = {"id": "a", "group": "yelp", "label": "positive"}
        b = {"id": "b", "group": "yelp", "label": "negative"}
        left = [a | {"prediction": "positive"}, b | {"prediction": "failure"}]
        right = [b | {"prediction": "negative"}, a | {"prediction": "invalid"}]
        self.assertEqual(wins(left, right), {"left_only": ["a"], "right_only": ["b"]})


if __name__ == "__main__":
    unittest.main()
