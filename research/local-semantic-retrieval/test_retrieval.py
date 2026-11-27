"""Check vector validity, deterministic ranking and retrieval metrics."""

import unittest
from evaluate_retrieval import bm25, metrics, normalize, rank_vectors


class RetrievalTests(unittest.TestCase):
    def test_rejects_invalid_vectors(self):
        for values in ([0,0],[float("nan"),1],[float("inf"),1],[1]):
            with self.subTest(values=values),self.assertRaises(ValueError):
                normalize(values,2)
        self.assertAlmostEqual(sum(x*x for x in normalize([3,4],2)),1,places=6)

    def test_ranking_and_ties_are_deterministic(self):
        self.assertEqual(rank_vectors(["b","a","c"],[[1,0],[1,0],[0,1]],[1,0]),
                         [("a",1),("b",1),("c",0)])
        with self.assertRaises(ValueError):
            rank_vectors(["a"],[[1,2]],[1])

    def test_top_k_and_reciprocal_rank(self):
        rows=[{"relevant":["a"],"ranking":["a","b","c","d"]},
              {"relevant":["c"],"ranking":["a","b","c","d"]},
              {"relevant":["d"],"ranking":["a","b","c","d"]}]
        result=metrics(rows)
        self.assertEqual(result["hit_at_1"],1/3)
        self.assertEqual(result["hit_at_3"],2/3)
        self.assertAlmostEqual(result["mrr"],(1+1/3+1/4)/3)

    def test_lexical_baseline_finds_unique_term(self):
        docs=[{"id":"a","title":"Policy","text":"refund receipt"},
              {"id":"b","title":"Policy","text":"password login"}]
        self.assertEqual(bm25(docs,"password",1.5,0.75)[0][0],"b")


if __name__=="__main__":
    unittest.main()
