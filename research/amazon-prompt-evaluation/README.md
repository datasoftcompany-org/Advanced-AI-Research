# Amazon sentiment prompt evaluation

## Research question

How does the direct sentiment prompt used in phases two and three perform on Amazon product reviews? Does instructing the same model to weigh overall judgment, negation, and contrast change its predictions?

## Protocol

We froze the protocol and sample before inference in commit `2732171`. The sample contains 60 positive and 60 negative Amazon product-review sentences from [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), Kotzias et al. (2015), distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The evaluator verifies the original archive hash before reading it. It applies the shared deduplication and class-stratified hash ranking. The remaining split has 436 negative and 431 positive labels, so the majority baseline always predicts negative.

Both prompts use the same local `llama3.2:3b` digest, exact-label rule, generation settings, sample and row order. Each prompt receives one unscored warm-up per round. The model is unloaded after each pass, and the second round reverses prompt order. The latency measurement includes HTTP and generation and excludes the warm-up and request failures.

[Protocol](protocol.json) · [Frozen sample](sample.json) · [Predictions and timings](results.json)

## Reproduce

Use Python 3.13 and a running local Ollama service with `llama3.2:3b` installed. The recorded setup used Python 3.13.4 on arm64 and Ollama 0.40.16. Run from the repository root.

```sh
curl -fL 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip' -o /tmp/sentiment-331.zip
python3 -B research/amazon-prompt-evaluation/compare_prompts.py --archive /tmp/sentiment-331.zip
```

Compare the output with the published JSON using the sample hashes, model digest and per-example predictions. Timings can vary. Run the regression test with `python3 -B -m unittest discover -s research/amazon-prompt-evaluation`.

## Results

Each prompt produced the same predictions in both rounds. Neither had request failures.

| Condition | Accuracy | Macro-F1 | Invalid per round | Median latency, rounds 1 / 2 |
| --- | ---: | ---: | ---: | ---: |
| Majority baseline | 60/120 (50.0%) | 0.333 | 0 | — |
| Direct prompt | 115/120 (95.8%) | 0.958 | 0 | 62.1 / 61.9 ms |
| Overall-judgment prompt | 104/120 (86.7%) | 0.869 | 1 | 61.2 / 61.3 ms |

The direct prompt alone correctly classified 12 examples. The overall-judgment prompt alone correctly classified one. The latter got one more negative example right in aggregate (58 versus 57), but correctly identified only 46 positives versus 58 for the direct prompt. It predicted negative for 13 positives and returned an invalid label for one positive. The JSON report records the confusion matrices and all paired outcomes by source row ID.

The direct prompt retained high accuracy in this additional domain. This is a descriptive comparison on one small balanced sample with a single model digest. The public data may have appeared in model training. Several instructions changed together between prompts, so the experiment cannot isolate the effect of mentioning negation or contrast.
