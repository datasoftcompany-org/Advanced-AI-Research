"""Check sample isolation and paired prediction alignment."""

import io
import unittest
import zipfile

from compare import align, load_sample, sha256


class ComparisonTests(unittest.TestCase):
    def test_excluded_rows_do_not_enter_either_split(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("fixture.tsv", "".join(
                f"Example {i}\t{i % 2}\n" for i in range(20)
            ))
        data = stream.getvalue()
        previous = {
            "dataset_sha256": sha256(data), "member": "fixture.tsv",
            "split_salt": "first", "test_per_label": 2,
        }
        test, training, excluded = load_sample(
            data, previous, {"split_salt": "second", "test_per_label": 3}
        )
        self.assertEqual(len(excluded), 4)
        self.assertEqual(len(test), 6)
        self.assertFalse(set(excluded) & {row["id"] for row in test})
        self.assertEqual(dict(training), {"positive": 5, "negative": 5})

    def test_pairing_uses_ids_and_rejects_inconsistent_results(self):
        a = {"id": "a", "group": "test", "label": "positive"}
        b = {"id": "b", "group": "test", "label": "negative"}
        self.assertEqual(align([a, b], [b, a]), [(a, a), (b, b)])
        for invalid in ([a], [a, a], [a, b | {"label": "positive"}]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                align([a, b], invalid)


if __name__ == "__main__":
    unittest.main()
