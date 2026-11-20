"""Guard multiline parsing and incomplete record handling."""

import io
import unittest
import zipfile

from evaluate_imdb import read_split, sha256


class MultilineTests(unittest.TestCase):
    def fixture(self, text):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("imdb.txt", text)
        data = stream.getvalue()
        return data, {
            "dataset_sha256": sha256(data), "member": "imdb.txt", "row_prefix": "imdb",
            "multiline_records": True, "split_salt": "test", "test_per_label": 1,
        }

    def test_continuations_stay_in_one_review(self):
        data, settings = self.fixture(
            "The script is\nexcellent\t1\nAnother good movie\t1\nThird good movie\t1\nBad\t0\nAwful\t0\nTerrible\t0\n"
        )
        test, counts = read_split(data, settings | {"test_per_label": 2})
        self.assertEqual(len(test), 4)
        # Choose a salt that puts the multiline row in the fixed test split.
        for salt in range(100):
            test, counts = read_split(data, settings | {"test_per_label": 2, "split_salt": str(salt)})
            if any(row["id"] == "imdb:2" for row in test):
                break
        row = next(row for row in test if row["id"] == "imdb:2")
        self.assertEqual(row["text"], "The script is\nexcellent")
        self.assertEqual(row["label"], "positive")
        self.assertEqual(sum(counts.values()), 2)

    def test_unterminated_review_is_rejected(self):
        data, settings = self.fixture("Missing label")
        with self.assertRaisesRegex(ValueError, "Unterminated"):
            read_split(data, settings)


if __name__ == "__main__":
    unittest.main()
