"""Guard PCM validity and avoid treating silence as successful speech."""

import unittest
import struct
from evaluate_speech import pcm_stats


class SpeechTests(unittest.TestCase):
    def test_silence_and_duration(self):
        stats = pcm_stats(b"\x00\x00" * 24000, 24000)
        self.assertEqual(stats["duration_seconds"], 1)
        self.assertEqual(stats["rms"], 0)
        self.assertFalse(stats["non_silent"])

    def test_peak_clipping_and_partial_samples(self):
        stats = pcm_stats(struct.pack("<hhh", 100, -32768, 32767), 24000)
        self.assertEqual(stats["peak_absolute"], 32768)
        self.assertEqual(stats["full_scale_samples"], 2)
        self.assertTrue(stats["non_silent"])
        for raw in (b"", b"x"):
            with self.assertRaises(ValueError):
                pcm_stats(raw, 24000)


if __name__ == "__main__":
    unittest.main()
