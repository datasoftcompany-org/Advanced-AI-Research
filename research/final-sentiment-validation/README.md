# Final held-out sentiment validation

## Objective and frozen choices

This closes the seven-phase local sentiment evaluation series. Phase six selected the direct prompt for Llama 3.2 3B and suggested checking Qwen 2.5 7B with negation and contrast on a fresh sample. We froze those choices, the new sample, confidence method and adoption rule before inference in commit `9f03627`.

The primary comparison is Qwen with negation and contrast minus Qwen direct on the first round's overall accuracy. We adopt the added instruction only if the lower endpoint of its paired 95% interval exceeds zero. Otherwise Qwen keeps direct. Llama keeps direct. Round two measures stability and contributes no additional independent observations to the confidence calculation.

## Data and method

The sample has 180 reviews, 30 per label in each of Yelp, Amazon and IMDb, from [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), Kotzias et al. (2015), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

We reuse the phase-six reader and selection method, adding the phase-six sample to the previous exclusion manifests. All 564 earlier UCI test texts are excluded after normalizing case and whitespace. The 180 selected texts are mutually distinct after that normalization. Authored diagnostic cases from earlier phases are outside these source-corpus exclusion manifests. The [sample](sample.json) records source hashes, exclusion hashes, text hashes, labels and remaining training counts.

Models and settings are fixed in the [protocol](protocol.json). Conditions run sequentially with one unscored warm-up and unload after each condition. The second round reverses condition order. Example and domain order stay fixed. Timings include HTTP and generation and exclude warm-up and failures. Context and output limits remain 2048 and eight tokens; truncation was not measured.

Confidence intervals use 5000 paired bootstrap resamples, seed 20261121, resampling jointly within the six domain-by-label strata. The 2.5th and 97.5th percentiles are linearly interpolated. We report selected-minus-direct accuracy and Macro-F1 differences. Domain intervals are exploratory and use the same method within that domain. There is one prespecified adoption comparison on overall accuracy.

## Reproduce

The recorded environment used Python 3.13.4 on arm64 and Ollama 0.40.16 with both models installed locally. From the repository root:

```sh
curl -fL 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip' -o /tmp/sentiment-331.zip
python3 -B research/final-sentiment-validation/validate.py --archive /tmp/sentiment-331.zip
```

Compare sample hashes, model digests, predictions and confidence intervals with [results.json](results.json). Timings vary with machine conditions. Test the confidence calculations with `python3 -B -m unittest discover -s research/final-sentiment-validation -v`.

## Final results

Predictions were identical between rounds for each condition. All 1080 scored requests completed with valid labels and no failures. The majority baseline scored 30/60 with Macro-F1 of 0.333 in each domain.

| Condition | Yelp /60 | Amazon /60 | IMDb /60 | Overall /180 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama direct | 59 | 60 | 58 | 177 | 98.3% | 0.983 |
| Qwen direct | 58 | 60 | 58 | 176 | 97.8% | 0.978 |
| Qwen negation and contrast | 57 | 60 | 57 | 174 | 96.7% | 0.967 |

The Qwen instruction addition changed two previously correct positive examples into negative predictions, one each in Yelp and IMDb, and corrected none of the direct prompt's errors.

| Selected minus direct | Estimate | Paired 95% bootstrap interval |
| --- | ---: | ---: |
| Overall accuracy | −1.11 percentage points | [−2.78, 0.00] percentage points |
| Overall Macro-F1 | −0.0111 | [−0.0279, 0.0000] |
| Yelp accuracy | −1.67 percentage points | [−5.00, 0.00] percentage points |
| Amazon accuracy | 0.00 percentage points | [0.00, 0.00] percentage points |
| IMDb accuracy | −1.67 percentage points | [−5.00, 0.00] percentage points |

The primary interval fails the prespecified adoption rule. The default prompt is therefore **direct for both models**. The exact direct prompt is in the protocol.

### Warmed median request latency

| Condition | Yelp ms, rounds 1 / 2 | Amazon ms, rounds 1 / 2 | IMDb ms, rounds 1 / 2 |
| --- | ---: | ---: | ---: |
| Llama direct | 86.4 / 85.2 | 67.8 / 70.1 | 88.9 / 89.7 |
| Qwen direct | 170.4 / 170.6 | 116.4 / 124.4 | 171.8 / 171.9 |
| Qwen negation and contrast | 170.8 / 171.2 | 121.0 / 123.9 | 171.9 / 172.5 |

## Interpretation and limits

Qwen's one-example gain in phase six did not repeat on the final sample. These results support retaining the direct prompt as the documented default in this local evaluation setting.

The bootstrap resamples this observed, balanced sample. It does not account for unknown training-data overlap, all possible review populations, other seeds or hardware variation. In particular, the zero-width Amazon interval arises because the paired predictions were identical on the observed examples; it does not establish population equivalence. Two repeated rounds verify stability on this setup and do not double the sample size. The one-example overall difference between models has no corresponding prespecified model-superiority test.
