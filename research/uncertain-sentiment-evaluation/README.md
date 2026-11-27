# Neutrality, ambiguity and human-review deferral

## Question and frozen study

Can an explicit uncertain option avoid forced polar classifications on neutral, equally mixed or insufficient opinion evidence while retaining correct decisions on clear sentiment?

The 80 authored examples and reference decisions were fixed before inference in commit `35d656d`. There are 20 positive, 20 negative, 10 neutral, 10 equally mixed and 20 insufficient-evidence cases. Ten product nouns each use eight shared sentence patterns. References request uncertain for the 40 cases outside clear polar sentiment. This definition treats uncertainty as a workflow decision to request human review.

[Cases and references](cases.json) · [Exact prompts](protocol.json) · [Frozen input hashes](manifest.json) · [Per-example evidence](results.json)

## Method

Both models use the original binary prompt and a second prompt permitting the exact labels positive, negative and uncertain, with instructions for neutral, insufficient and balanced opinions. The accepted label set changes only when explicitly supplied to the shared predictor; its default binary behavior remains covered by regression tests.

The same 80 cases appear in the same order in every condition. Models and prompt conditions run sequentially, with an unscored warm-up, unload between conditions and reversed condition order in round two. Settings retain seed 42, temperature zero, 2048 context tokens and eight output tokens. Truncation was not measured. Python 3.13.4, arm64 and Ollama 0.40.16 were recorded.

Metrics include clear-case accuracy (referrals and malformed answers count as errors), ambiguous referral rate, false referrals on clear cases, polar coverage across all cases and separate invalid/failure counts. Median request time includes each scored request. Forced binary responses on ambiguous examples are counted as forced decisions. This is a comparison of workflow tradeoffs between different available label sets.

## Results

Predictions repeated exactly between rounds. There were no request failures in 640 scored requests.

| Model and prompt | Clear correct /40 | Ambiguous referred /40 | Clear false referrals | Ambiguous forced polar decisions | Invalid per round |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama binary | 40 | 0 | 0 | 40 | 0 |
| Llama with uncertain | 40 | 20 | 0 | 20 | 0 |
| Qwen binary | 40 | 0 | 0 | 23 | 17 |
| Qwen with uncertain | 40 | 40 | 0 | 0 | 0 |

Llama's uncertain condition referred all 10 mixed cases and 10 of 20 insufficient cases, but none of the 10 neutral cases. Qwen's uncertain condition referred every neutral, mixed and insufficient case. Neither model referred a clear positive or negative case.

Qwen's 17 binary-mode invalid responses are recorded as invalid rather than intentional referrals. The shared predictor does not retain their raw text, so the record cannot identify their intended meaning.

| Model and prompt | Polar coverage | Median request time, rounds 1 / 2 |
| --- | ---: | ---: |
| Llama binary | 100.0% | 90.7 / 90.6 ms |
| Llama with uncertain | 75.0% | 90.7 / 88.5 ms |
| Qwen binary | 78.8% | 173.0 / 173.7 ms |
| Qwen with uncertain | 50.0% | 178.2 / 176.6 ms |

## Recommendation and limits

The explicit uncertain option is a candidate for an opt-in human-review mode with Qwen. Its success here supports further evaluation with independently annotated neutral and mixed text before adopting it as a product default. Llama's neutral-case misses need particular attention if it is considered for that mode.

The examples deliberately state clear neutrality or balanced opinions and share sentence templates. Their effective linguistic diversity is much smaller than 80 independently collected cases. Labels were authored without independent annotation, and there are no borderline clear-case examples designed to estimate over-referral in natural usage. The uncertain response is not a calibrated confidence estimate. The 100% clear-case scores and Qwen's full referral rate describe this fixture.

The local tool's binary default is maintained. This study adds the research evaluator and an opt-in accepted-label argument without changing the tool's UI or export meanings.

## Reproduce

With both models installed locally and Ollama running, execute from the repository root:

```sh
python3 -B research/uncertain-sentiment-evaluation/evaluate_uncertain.py
```

Compare model digests, input hashes and per-example predictions with the committed JSON. Timings depend on the local setup. The evaluator rejects modified inputs against the frozen manifest.

```sh
python3 -B -m unittest discover -s research/uncertain-sentiment-evaluation -v
```
