"""Check source integrity, binary filtering and duplicate handling."""

import tempfile
from pathlib import Path
import unittest

from evaluate_source import build_sample, sha256


class SourceTests(unittest.TestCase):
    def fixture(self, directory):
        files = {
            "test_text.txt": "Good\nGOOD\nBad\nFine\nPoor\nNeutral\nConflict\nconflict\n",
            "test_labels.txt": "2\n2\n0\n2\n0\n1\n0\n2\n",
            "train_labels.txt": "2\n2\n0\n1\n",
            "mapping.txt": "0\tnegative\n1\tneutral\n2\tpositive\n",
        }
        for name, text in files.items():
            (directory / name).write_text(text)
        return {"file_hashes": {name: sha256((directory / name).read_bytes()) for name in files},
                "salt": "test", "per_label": 2}

    def test_binary_selection_drops_neutral_and_conflicting_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            protocol = self.fixture(directory)
            rows, manifest = build_sample(directory, protocol)
        self.assertEqual(len(rows), 4)
        self.assertEqual(manifest["eligible_binary_records"], 4)
        self.assertEqual({row["label"] for row in rows}, {"negative", "positive"})
        self.assertNotIn("tweeteval:test:6", [row["id"] for row in rows])
        self.assertEqual(manifest["training_counts"], {"2": 2, "0": 1, "1": 1})

    def test_hash_mismatch_stops_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            protocol = self.fixture(directory)
            (directory / "test_text.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_sample(directory, protocol)

    def test_alignment_and_label_errors_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            protocol = self.fixture(directory)
            for content, message in (("0\n", "counts differ"), ("9\n" * 8, "Unknown")):
                path = directory / "test_labels.txt"
                path.write_text(content)
                protocol["file_hashes"][path.name] = sha256(path.read_bytes())
                with self.assertRaisesRegex(ValueError, message):
                    build_sample(directory, protocol)


if __name__ == "__main__":
    unittest.main()
