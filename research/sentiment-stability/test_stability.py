"""Distinguish consistency, correctness, unusable replies and missing pairs."""

import unittest

from evaluate_stability import examples_from, paired_summary


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{"id": "a", "label": "positive", "original": "Good.",
                       "variants": {"paraphrase": "Enjoyable.", "typo": "Goood.", "punctuation": "Good!"}}]

    def rows(self, original, variant):
        return [row | {"prediction": original if row["group"] == "original" else variant}
                for row in examples_from(self.cases)]

    def test_agreement_is_not_necessarily_correct(self):
        score = paired_summary(self.cases, self.rows("negative", "negative"))["typo"]
        self.assertEqual(score["valid_agreement"], 1)
        self.assertEqual(score["both_correct"], 0)
        self.assertEqual(score["correct_to_wrong"], 0)

    def test_errors_are_not_stable_and_correctness_flips_are_counted(self):
        score = paired_summary(self.cases, self.rows("failure", "failure"))["typo"]
        self.assertEqual(score["valid_agreement"], 0)
        self.assertEqual(score["either_unusable"], 1)
        score = paired_summary(self.cases, self.rows("positive", "invalid"))["typo"]
        self.assertEqual(score["correct_to_wrong"], 1)
        score = paired_summary(self.cases, self.rows("negative", "positive"))["typo"]
        self.assertEqual(score["wrong_to_correct"], 1)

    def test_missing_and_duplicate_predictions_are_rejected(self):
        rows = self.rows("positive", "positive")
        for bad in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                paired_summary(self.cases, bad)


if __name__ == "__main__":
    unittest.main()
