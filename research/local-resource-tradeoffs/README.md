# Local quality, latency and allocation tradeoffs

## Scope and fixed benchmark

This is the fourth and final stage of the new research series. It compares the two installed model sizes at context lengths 1024 and 2048 using the approved direct prompt.

The protocol and source hashes were frozen before inference in `1e6f4e8`. The fixed set contains the 60 authored acceptance reviews and 24 spelling variants from the stability study: 84 texts, balanced by sentiment. These examples and their outcomes have already been examined in earlier studies. This benchmark compares resources under known workloads; its accuracy is not a fresh population estimate.

[Protocol](protocol.json) · [Source hashes](manifest.json) · [Predictions, digests and allocation snapshots](results.json)

## Measurement

Conditions run sequentially, then in reverse order in round two. Each condition starts with the model unloaded. Its first unscored generation measures model load plus a fixed warm-up. The operating-system cache is not cleared between runs. Scored request times include HTTP and generation, excluding the allocation queries between requests.

Median and nearest-rank 95th-percentile times exclude request failures. Quality metrics count invalid responses and failures as errors. The benchmark checks model digests and service versions.

After warm-up and each scored request, the benchmark samples Ollama's /api/ps fields size, size_vram and context_length for the target model. It reports the maximum observed size and size_vram. These are loaded-model allocation figures reported by Ollama, rather than measured whole-process resident memory or instantaneous peak usage. On Apple unified memory, the GPU figure and size overlap and should not be added.

The machine reported 18 GiB physical RAM, arm64, macOS 27.2, Python 3.13.4 and Ollama 0.40.16. Power consumption and system-wide memory pressure were outside measurement scope.

## Results

All 672 scored requests returned valid labels with no failures. Predictions were identical across reversed rounds and between context lengths for each model.

| Model / context | Correct /84 | Macro-F1 | Median ms, rounds 1 / 2 | P95 ms, rounds 1 / 2 | Max reported size GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama / 1024 | 76 | 0.904 | 67.9 / 68.5 | 97.2 / 98.4 | 2.05 |
| Llama / 2048 | 76 | 0.904 | 67.6 / 70.0 | 96.4 / 97.9 | 2.16 |
| Qwen / 1024 | 82 | 0.976 | 132.1 / 134.0 | 182.5 / 187.4 | 4.26 |
| Qwen / 2048 | 82 | 0.976 | 132.5 / 133.6 | 187.6 / 187.4 | 4.32 |

Ollama reported size_vram equal to size in every sampled allocation. All sampled context_length values matched the requested condition. Reducing context saved 113 MiB for Llama and 57 MiB for Qwen, with unchanged predictions on these short texts. Observed model-load-plus-warm-up times ranged from 1.00 to 2.03 seconds for Llama and 2.76 to 4.50 seconds for Qwen.

Llama scored 56/60 on acceptance reviews and 20/24 on spelling variants. Qwen scored 58/60 and 24/24 respectively. On this deliberately spelling-stressed workload, Qwen gained six correct classifications with roughly twice the median request latency and loaded allocation.

## Operational recommendation

Keep the direct prompt for both models. Llama with the existing 2048-token setting remains the documented lightweight starting option. Qwen with 2048 is the quality-oriented option when spelling tolerance or ambiguous text warrants its measured resource cost. The explicit uncertain workflow remains a research candidate from stage two.

A 1024-token context is a possible constrained-memory profile for similarly short inputs. Its modest allocation savings and unmeasured truncation behavior on longer inputs do not justify replacing the existing 2048 setting from this experiment alone.

Users should review consequential predictions and assess the intended domain. The independent-source study found lower accuracy on social posts, and the neutral/ambiguous results came from a small authored fixture. This stage completes the comparison and recommendation without changing application defaults.

## Reproduce

Use the existing local models with Ollama running and no models loaded at the start. Run from the repository root:

```sh
python3 -B research/local-resource-tradeoffs/benchmark.py
python3 -B -m unittest discover -s research/local-resource-tradeoffs -v
```

The runner verifies all source hashes and rejects an occupied Ollama instance at startup. It unloads each target after its condition. Compare identities, examples, scores and allocation snapshots with the saved JSON. Latency and allocation can vary with machine state and Ollama versions.
