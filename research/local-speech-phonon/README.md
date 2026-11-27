# Local Phonon speech generation

## Scope and frozen inputs

This is the first practical stage of the local capability research scope. It tests the installed `phonon:latest` model with the `default` voice on ten English texts repeated twice. Inputs and measurement settings were committed before generation in `2d62cb3`.

[Cases](cases.json) cover a greeting, spelled-out numbers, currency, abbreviations, written and numeric dates, punctuation, negation, contact details and multiline continuity. [Protocol](protocol.json) · [Input hashes](manifest.json) · [Recorded measurements](results.json)

No model was downloaded. All inference used the local Ollama endpoint. The recorded model digest is `58f58601e4e83c6d5c06f87ca91071e0a9acfd6d169515fd8eb3da27965b990e`. The local metadata reports a 43.49M bfloat16 Gradium Phonon model with an MLX runner. This identifies the actual installed package rather than assuming that a mutable latest tag has the current registry's architecture.

## Method

The local OpenAI-compatible speech endpoint returns raw PCM with the requested default voice. The runner buffers 1024-byte reads, measures delivery time to the first non-empty chunk and completion, and wraps the PCM in standard WAV headers using Python's wave module. Output is 24 kHz, mono, 16-bit little-endian PCM. [Ollama speech documentation](https://docs.ollama.com/capabilities/speech)

First-chunk time is client-observed delivery of up to 1024 PCM bytes. It includes request overhead and buffering. It does not measure the onset of audible speech within the waveform. Real-time factor is complete request time divided by output duration; lower values indicate faster generation.

The runner checks local model identity before and after the run, writes only to a new output directory, records hashes and retains checkpoint results after each completed case. WAV files are stored locally outside the public repository to conserve transfer volume.

## Technical results

Recorded setup: Python 3.13.4, arm64, Ollama 0.40.16. All 20 requests produced readable WAV files with nonzero RMS. Channel count, sample width, sample rate, frame count and PCM hashes were checked. No full-scale samples were observed.

| Measurement | Round 1 | Round 2 |
| --- | ---: | ---: |
| Successful files | 10/10 | 10/10 |
| Median first-chunk delivery | 9.5 ms | 9.2 ms |
| First-chunk range | 8.3–441.3 ms | 7.5–13.7 ms |
| Median real-time factor | 0.0225 | 0.0220 |

Total output duration was 113.76 seconds. Summed generation request time was 3.1545 seconds. The first greeting request took 0.564 seconds in total, including any model loading. The contact-details case had an additional first-round delay; the experiment does not establish its cause.

| Case | Output duration, rounds 1 / 2 |
| --- | ---: |
| Greeting | 3.84 / 3.76 s |
| Counts | 3.76 / 3.84 s |
| Prices | 6.24 / 6.56 s |
| Abbreviations | 5.20 / 5.36 s |
| Written date | 4.64 / 4.56 s |
| Numeric date | 3.52 / 3.60 s |
| Punctuation | 3.84 / 4.00 s |
| Negation | 4.88 / 4.80 s |
| Contact details | 8.48 / 8.24 s |
| Paragraph | 12.08 / 12.56 s |

None of the ten paired PCM hashes matched across rounds. The measurements establish successful local generation and delivery; the current experiment does not assume deterministic waveform output.

## Listening review

Pronunciation, intelligibility, missing or added words, pauses, and naturalness remain pending human listening review. Check numbers and prices against the exact text, how abbreviations and contact details are spoken, and whether negation and multiline sentences remain intelligible. Waveform validity, nonzero RMS and the absence of full-scale samples do not establish semantic accuracy or perceptual quality.

The installed package reports multiple available voices, recorded in the JSON. Only default was tested. Other voices, languages, long inputs, concurrent use and end-to-end playback latency are outside this run.

## Reproduce locally

With the model already installed and Ollama running, run from the repository root:

```sh
python3 -B research/local-speech-phonon/evaluate_speech.py --output /tmp/phonon-new-run
python3 -B -m unittest discover -s research/local-speech-phonon -v
```

Choose a new output directory for each run. Open the resulting WAV files with a local audio player. The completed local run is stored at `/Users/m3/Documents/AI-Research-Outputs/phonon-local-01`; it contains both rounds and a measurement JSON. The public repository contains text and measurements only. Generation speed varies with model state and machine load.
