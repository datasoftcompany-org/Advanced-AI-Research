# Incremental sentiment prompt instructions

## Research question

How do incremental instructions about negation, contrast, and overall judgment affect two local models on fresh Yelp, Amazon, and IMDb examples? This evaluation removes the product-specific wording that complicated the phase-five comparison.

## Frozen protocol and sample

The protocol, runner, and sample were committed before inference in `9ef2006`. The [protocol](protocol.json) records the exact four prompts. Each step changes the preceding prompt as follows.

| Condition | Change |
| --- | --- |
| Direct | Original direct prompt from phase two |
| Negation | Insert “Consider negation.” |
| Negation and contrast | Extend that instruction to “Consider negation and contrast.” |
| Overall | Add “Choose the label that best reflects the writer's overall judgment.” |

The prompts share the same exact-label instruction and JSON review delimiter. None names products, restaurants, or movies. We retain temperature 0, seed 42, a 2048-token context and eight-token generation limit. Truncation was not measured.

Data comes from [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), Kotzias et al. (2015), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The archive hash is verified before reading. Each of the three domains contributes 30 positive and 30 negative reviews, for 180 examples in total. Normalized case and whitespace identify consistently labelled duplicates; conflicting duplicate groups are dropped. IMDb continuation lines remain part of their review.

We reconstruct the 384 prior UCI test examples from the phase-two results and phase-three, four, and five sample manifests. Their normalized text hashes are excluded across all domains. We also exclude already selected text when selecting a later domain. The historical authored diagnostic cases are outside these source-corpus exclusion manifests. Within each class, SHA-256 of the fixed salt and row ID determines selection. The [sample manifest](sample.json) records exclusions, source hashes, selected text hashes, labels, and remaining training counts.

Each condition is evaluated sequentially on all 180 examples. The second round reverses the order of all eight conditions. Each condition has one unscored warm-up and unloads the model afterward. Domain and example order remain fixed. Timing includes HTTP and generation and excludes warm-up and failed requests. The majority baseline is determined separately from the remaining labels in each domain.

## Reproduce

Use Python 3.13 and a local Ollama instance with `llama3.2:3b` and `qwen2.5:7b` already installed. The recorded setup used Python 3.13.4 on arm64 and Ollama 0.40.16.

```sh
curl -fL 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip' -o /tmp/sentiment-331.zip
python3 -B research/prompt-instruction-ablation/evaluate_ablation.py --archive /tmp/sentiment-331.zip
```

The runner verifies the regenerated sample against the committed manifest. Compare model digests and per-row predictions with [results.json](results.json). Reproduction of timings depends on local load and hardware. Run regression tests with `python3 -B -m unittest discover -s research/prompt-instruction-ablation -v`.

## Results

All eight conditions produced identical predictions in the two rounds. There were zero invalid labels and zero request failures across 2880 scored requests. The baseline scored 30/60 with Macro-F1 of 0.333 in each domain.

| Model | Prompt | Yelp /60 | Amazon /60 | IMDb /60 | Total /180 | Overall Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Llama | Direct | 58 | 59 | 60 | 177 | 0.983 |
| Llama | Negation | 58 | 59 | 59 | 176 | 0.978 |
| Llama | Negation and contrast | 58 | 59 | 59 | 176 | 0.978 |
| Llama | Overall | 56 | 56 | 55 | 167 | 0.927 |
| Qwen | Direct | 58 | 58 | 58 | 174 | 0.967 |
| Qwen | Negation | 58 | 58 | 57 | 173 | 0.961 |
| Qwen | Negation and contrast | 58 | 59 | 58 | 175 | 0.972 |
| Qwen | Overall | 58 | 58 | 59 | 175 | 0.972 |

### Changes between adjacent prompts

For Llama, the negation instruction introduced one additional error. Adding contrast changed no predictions. Adding overall judgment introduced nine additional errors without correcting an earlier error.

For Qwen, negation corrected one example and introduced two errors. Adding contrast corrected two errors without introducing a new error. Adding overall judgment corrected one example and introduced one error, leaving total accuracy unchanged. The JSON contains the IDs for every exclusive correct prediction and the domain-level confusion matrices.

### Warmed request latency

| Model and condition | Yelp median ms, rounds 1 / 2 | Amazon median ms, rounds 1 / 2 | IMDb median ms, rounds 1 / 2 |
| --- | ---: | ---: | ---: |
| Llama direct | 82.7 / 87.1 | 75.1 / 82.3 | 86.0 / 94.5 |
| Llama negation | 83.4 / 87.8 | 75.9 / 77.5 | 84.7 / 89.9 |
| Llama negation and contrast | 84.2 / 86.6 | 78.1 / 81.9 | 84.5 / 91.0 |
| Llama overall | 83.0 / 88.4 | 75.4 / 86.3 | 84.6 / 90.6 |
| Qwen direct | 171.8 / 171.8 | 171.0 / 171.4 | 173.4 / 172.1 |
| Qwen negation | 171.6 / 171.2 | 171.1 / 170.3 | 172.8 / 176.0 |
| Qwen negation and contrast | 171.4 / 172.2 | 170.4 / 170.7 | 175.4 / 172.5 |
| Qwen overall | 171.3 / 171.6 | 171.7 / 171.1 | 172.4 / 174.9 |

## Interpretation and limits

The direct prompt was best for Llama on this sample. Qwen gained one correct answer overall with the contrast instruction sequence compared with direct, while performance differed by domain. The neutral overall instruction still reduced Llama's accuracy, so product-specific wording alone cannot account for all earlier reductions. These comparisons identify sensitivity to the tested wording and order of additions.

This sample was used for comparing prompts and cannot serve as an untouched final validation set for a prompt chosen from these results. Small net differences, such as one extra correct example, need further evaluation. The experiment uses one fixed seed, two model digests and one public dataset collection. Potential overlap with model pretraining is unknown. Runtime load, fixed example order and the short output limit can affect observations. The report supports descriptive comparisons on the recorded sample.
