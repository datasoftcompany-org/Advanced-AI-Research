"""Checks for resource attribution and percentile reporting."""

import unittest

from benchmark import allocation, examples_from, percentile95


class BenchmarkTests(unittest.TestCase):
    def test_allocation_is_attributed_to_target_model(self):
        records = [
            {"name": "other", "size": 900, "size_vram": 800},
            {"name": "target", "size": 200, "size_vram": 100, "context_length": 1024},
        ]
        self.assertEqual(allocation(records, "target"),
                         {"size": 200, "size_vram": 100, "context_length": 1024})
        with self.assertRaises(ValueError):
            allocation(records, "missing")

    def test_unknown_gpu_allocation_stays_unknown(self):
        self.assertIsNone(allocation([{"model": "target", "size": 10}], "target")["size_vram"])

    def test_p95_is_nearest_rank(self):
        self.assertEqual(percentile95(list(range(1, 21))), 19)
        self.assertEqual(percentile95([2]), 2)

    def test_empty_acceptance_rows_are_excluded(self):
        raw = {
            "reviews": b'text\nGood\n""\n',
            "references": b'row,sentiment,status\n1,positive,ok\n2,,empty\n',
            "stability": b'[{"id":"a","label":"negative","variants":{"typo":"Baad"}}]',
        }
        rows = examples_from(raw)
        self.assertEqual([row["group"] for row in rows], ["acceptance", "spelling"])
        self.assertEqual([row["label"] for row in rows], ["positive", "negative"])


if __name__ == "__main__":
    unittest.main()
