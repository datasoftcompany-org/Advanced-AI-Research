"""Check prompt wins using matched examples."""

import unittest

from compare_prompts import paired_outcomes


class PairedOutcomesTests(unittest.TestCase):
    def test_matches_by_id_and_counts_invalid_predictions_as_incorrect(self):
        a = {"id": "a", "group": "test", "label": "positive"}
        b = {"id": "b", "group": "test", "label": "negative"}
        direct = [a | {"prediction": "positive"}, b | {"prediction": "positive"}]
        overall = [b | {"prediction": "negative"}, a | {"prediction": "invalid"}]
        self.assertEqual(paired_outcomes(direct, overall), {
            "direct_only": ["a"], "overall_only": ["b"],
        })
