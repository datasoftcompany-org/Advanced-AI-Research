"""Check tool boundaries, intermediate evidence and bounded model loops."""

import unittest
from unittest.mock import patch

from evaluate_tools import correct, dispatch, run_task


class ToolTests(unittest.TestCase):
    def test_rejects_arbitrary_tools_and_bad_arguments(self):
        for name, args in (
            ("shell", {"command": "anything"}),
            ("stock", {"sku": "A", "path": "/tmp"}),
            ("arithmetic", {"operation": "add", "left": True, "right": 1}),
            ("arithmetic", {"operation": "add", "left": float("nan"), "right": 1}),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                dispatch(name, args, {"A": 1})

    def test_expected_errors_are_tool_results(self):
        self.assertEqual(dispatch("stock", {"sku": "missing"}, {}), {"error": "unknown SKU"})
        self.assertEqual(dispatch("arithmetic", {"operation": "divide", "left": 1, "right": 0}, {}),
                         {"error": "division by zero"})

    def test_guessed_answer_without_correct_tool_evidence_fails(self):
        task = {"expected": 12, "required_tools": ["arithmetic"],
                "expected_tool_results": [{"value": 12}]}
        self.assertTrue(correct(task, 12, ["arithmetic"], 0, [{"value": 12}]))
        self.assertFalse(correct(task, 12, [], 0, []))
        self.assertFalse(correct(task, 12, ["arithmetic"], 0, [{"value": 5}]))
        self.assertFalse(correct(task, True, ["arithmetic"], 0, [{"value": 12}]))

    def test_call_limit_bounds_repeated_tool_calls(self):
        task = {"id": "loop", "prompt": "test"}
        config = {"system": "test", "tools": [], "options": {}, "max_turns": 6,
                  "max_calls": 2, "inventory": {}, "think_qwen3": False}
        response = {"done": True, "message": {"content": "", "tool_calls": [
            {"function": {"name": "stock", "arguments": {"sku": "missing"}}}
        ]}}
        with patch("evaluate_tools.request", return_value=response):
            result = run_task("qwen3:8b", task, config)
        self.assertEqual(result["outcome"], "call_limit")
        self.assertEqual(len(result["calls"]), 2)
        self.assertFalse(result["correct"])


if __name__ == "__main__":
    unittest.main()
