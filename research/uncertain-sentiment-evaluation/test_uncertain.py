"""Keep deferral separate from service and parsing errors."""

import unittest
from unittest.mock import patch

from evaluate_uncertain import predict, summarize


class DeferralTests(unittest.TestCase):
    def test_uncertain_requires_explicit_opt_in(self):
        settings = {"timeout_seconds": 1, "model": "test", "prompt": "",
                    "options": {}, "keep_alive": 0}
        with patch("evaluate.request", return_value={"done": True, "response": "uncertain"}):
            self.assertEqual(predict("text", settings), "invalid")
            self.assertEqual(predict("text", settings, ("positive", "negative", "uncertain")), "uncertain")

    def test_failures_never_count_as_intentional_deferrals(self):
        rows = [
            {"reference": "positive", "prediction": "uncertain", "category": "clear", "seconds": 1},
            {"reference": "negative", "prediction": "negative", "category": "clear", "seconds": 1},
            {"reference": "uncertain", "prediction": "failure", "category": "ambiguous", "seconds": 1},
            {"reference": "uncertain", "prediction": "invalid", "category": "ambiguous", "seconds": 1},
            {"reference": "uncertain", "prediction": "uncertain", "category": "ambiguous", "seconds": 1},
        ]
        scores = summarize(rows)
        self.assertEqual(scores["clear_accuracy"], 0.5)
        self.assertEqual(scores["clear_false_referrals"], 1)
        self.assertEqual(scores["ambiguous_referral_rate"], 1 / 3)
        self.assertEqual(scores["counts"]["failure"], 1)


if __name__ == "__main__":
    unittest.main()
