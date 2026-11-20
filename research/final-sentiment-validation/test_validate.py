"""Guard paired confidence intervals and percentile endpoints."""

import unittest

from validate import paired_intervals, percentile


class ConfidenceTests(unittest.TestCase):
    settings = {"seed": 42, "replicates": 200, "confidence": 0.95}

    def test_identical_predictions_have_zero_paired_difference(self):
        rows = [{"id": str(i), "group": "a", "label": label, "prediction": label}
                for i, label in enumerate(("positive", "negative", "positive", "negative"))]
        result = paired_intervals(rows, list(reversed(rows)), self.settings)
        for scores in result.values():
            self.assertEqual(scores["selected_minus_direct"], 0)
            self.assertEqual(scores["interval"], [0, 0])

    def test_improvement_direction_and_seed_reproducibility(self):
        left = [{"id": str(i), "group": "a", "label": label, "prediction": "failure"}
                for i, label in enumerate(("positive", "negative"))]
        right = [row | {"prediction": row["label"]} for row in left]
        result = paired_intervals(left, right, self.settings)
        self.assertEqual(result["accuracy"]["interval"], [1, 1])
        self.assertEqual(result["macro_f1"]["selected_minus_direct"], 1)
        self.assertEqual(result, paired_intervals(left, right, self.settings))

    def test_percentile_interpolates_and_accepts_endpoints(self):
        self.assertEqual(percentile([0, 10], 0), 0)
        self.assertEqual(percentile([0, 10], 1), 10)
        self.assertEqual(percentile([0, 10], 0.25), 2.5)


if __name__ == "__main__":
    unittest.main()
