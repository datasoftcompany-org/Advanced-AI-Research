"""Keep strict answers and failed requests visible in image scores."""

import unittest

from evaluate_vision import parse_answer, score


class VisionTests(unittest.TestCase):
    def test_only_completed_exact_labels_are_accepted(self):
        self.assertEqual(parse_answer({"done":True,"message":{"content":" yes\n"}}),"yes")
        self.assertEqual(parse_answer({"done":True,"message":{"content":"Yes, it is."}}),"invalid")
        self.assertEqual(parse_answer({"done":False,"message":{"content":"yes"}}),"failure")
        self.assertEqual(parse_answer({}),"failure")

    def test_unusable_answers_count_in_denominator(self):
        rows=[
            {"reference":"yes","prediction":"yes"},
            {"reference":"no","prediction":"yes"},
            {"reference":"yes","prediction":"invalid"},
            {"reference":"no","prediction":"failure"},
        ]
        result=score(rows)
        self.assertEqual(result["n"],4)
        self.assertEqual(result["correct"],1)
        self.assertEqual(result["counts"],{"yes":2,"invalid":1,"failure":1})


if __name__=="__main__":
    unittest.main()
