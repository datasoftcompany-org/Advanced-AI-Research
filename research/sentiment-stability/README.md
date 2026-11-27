# Sentiment stability under meaning-preserving edits

## Research question

Does sentiment classification remain consistent when an English review is paraphrased, contains a small spelling mistake, or changes terminal punctuation? This is stage three of the new research series.

## Fixture and method

The 24 original reviews, their reference labels and three variants per review were committed before inference in `ea5677b`. Twelve originals are positive and twelve are negative. Each has an authored paraphrase, a spelling-error variant, and a punctuation variant replacing the final period with an exclamation mark. There are 96 distinct input texts and 72 original-to-variant pairs per model.

The author checked that each edit preserves the intended sentiment before evaluation. Independent semantic annotation was not performed. The fixture includes direct opinions, negation, contrasts and implied approval or disapproval. Texts and references are in [cases.json](cases.json).

We use the existing direct prompt and fixed settings, with sequential requests, an unscored warm-up per model, and unload between model runs. Round two reverses model order while preserving example order. The recorded environment is Python 3.13.4 on arm64 and Ollama 0.40.16. Prompt and fixture hashes are fixed in [manifest.json](manifest.json). [Protocol](protocol.json) · [Predictions, model digests, metrics and pair records](results.json)

## Metrics

Valid agreement counts pairs with the same accepted positive or negative prediction. Matching invalid or failed responses do not count as stable. Both-correct counts pairs where both predictions match the reference. Correct-to-wrong counts loss of a correct original prediction after transformation, including unusable replies; wrong-to-correct counts the reverse. All pairs remain in the agreement-rate denominator.

Standard accuracy, Macro-F1, confusion matrices, invalid counts, failure counts and median request times are also reported per transformation. Variants of one review are dependent observations.

## Results

Both models repeated all their predictions in round two. All 384 scored requests returned valid labels without request failures.

| Input group | Llama correct /24 | Llama Macro-F1 | Qwen correct /24 | Qwen Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Original | 23 | 0.958 | 24 | 1.000 |
| Paraphrase | 24 | 1.000 | 24 | 1.000 |
| Spelling error | 20 | 0.829 | 24 | 1.000 |
| Punctuation | 24 | 1.000 | 24 | 1.000 |

| Transformation | Llama valid agreement /24 | Llama both correct /24 | Llama correct-to-wrong | Llama wrong-to-correct | Qwen valid agreement and both correct /24 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paraphrase | 23 | 23 | 0 | 1 | 24 |
| Spelling error | 21 | 20 | 3 | 0 | 24 |
| Punctuation | 23 | 23 | 0 | 1 | 24 |

The spelling edits in review:04 (“intrested”), review:09 (“beatifully”) and review:10 (“wroth”) changed Llama's correct positive predictions to negative. Review:11's original repeat-purchase statement was already misclassified negative. Its misspelled version was also negative, giving valid agreement without correctness. Paraphrasing it or replacing the final period with an exclamation mark produced the correct positive prediction.

Qwen correctly classified every original and variant in this fixture. This provides local evidence of greater stability on these authored examples. It does not establish robustness across arbitrary spelling errors or paraphrases.

Median warmed request times across the four groups ranged from 61.2 to 69.6 ms for Llama and from 117.7 to 134.2 ms for Qwen in round one. All individual timings and round-two medians are retained in the JSON.

## Implications and limitations

The local evaluation can report high baseline accuracy while small input changes introduce errors. Stability and correctness should be reviewed together. The spelling-error cases identify a specific weakness worth considering alongside runtime cost in the final resource comparison.

There are only 24 original reviews. Their variants do not constitute 72 independent natural reviews, and the punctuation test covers one narrow edit. Labels and semantic equivalence were checked by the author without independent annotation. The fixed seed, short generation limit and potential model training overlap limit generalization. This study does not change the application's default model or prompt.

## Reproduce

With both models installed and local Ollama running, execute from the repository root:

```sh
python3 -B research/sentiment-stability/evaluate_stability.py
python3 -B -m unittest discover -s research/sentiment-stability -v
```

The evaluator checks frozen input hashes before inference. Compare per-example predictions, model digests, group scores and pair records with the published result. Timings can vary.
