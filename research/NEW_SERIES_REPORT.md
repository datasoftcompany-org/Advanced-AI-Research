# Generalization, deferral, stability and local resources

The four-stage research series has completed its planned evaluations and operational recommendations.

| Stage | Evidence | Main result |
| --- | --- | --- |
| 1 | [Independent-source validation](independent-source-validation/README.md) | On 160 non-neutral TweetEval posts, Llama scored 137/160 and Qwen 140/160. Both were below the earlier review-sample scores. |
| 2 | [Neutrality and ambiguity](uncertain-sentiment-evaluation/README.md) | With an explicit uncertain option, Qwen referred 40/40 authored ambiguous cases and Llama 20/40. Both classified 40/40 clear examples correctly. |
| 3 | [Input stability](sentiment-stability/README.md) | Qwen correctly classified all 96 original and edited texts. Three spelling edits changed correct Llama predictions to errors. |
| 4 | [Quality and resources](local-resource-tradeoffs/README.md) | On 84 known texts, Qwen scored 82/84 and Llama 76/84, at roughly twice the measured median latency and loaded allocation. |

## Decision for local use

Retain the validated direct prompt. Llama with context 2048 is the lightweight starting configuration; Qwen with context 2048 is the quality-oriented choice when its resource cost is acceptable. The resource study supports that choice on its measured workload. The current application already allows choosing either model.

The explicit uncertain response is a candidate for a separately validated opt-in human-review mode. The strong Qwen result on authored ambiguous templates needs independent annotation and natural examples before adoption as a default.

## Evidence boundaries and closure

The independent-source stage evaluates social posts with neutral labels excluded. It does not establish full three-class sentiment performance. The later fixtures were authored, and repeated variants or templates reduce their effective diversity. Resource measurements refer to Ollama-reported allocation and local request times. Model pretraining overlap, actual energy use and whole-system peak memory remain unknown.

All protocols were frozen before their respective inference runs. Every stage provides hashes, model identities, per-example predictions and reproduction instructions. The resource stage reuses earlier reviewed examples explicitly for a fixed workload comparison.

This report closes the four-stage scope. Application feature changes, additional models, languages or deployment targets can be scoped as separate work.
