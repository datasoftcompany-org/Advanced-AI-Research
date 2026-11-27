# Research projects

Create a subdirectory under `research/` for each project accepted for this public repository. Use a short, lowercase name with hyphens. Use [PROJECT_TEMPLATE.md](PROJECT_TEMPLATE.md) for the project README.

Add source code, tests, and configuration as the project needs them. Record the exact runtime and dependency versions, model and dataset versions, and the command used to reproduce each result. Use a separate environment per project and commit its dependency manifest and lockfile when available.

Keep credentials, private datasets, unpublished material, large model weights, and generated runs outside the public repository. Record shareable dataset and model references in the project README.

## Project index

- [Sentiment classification baseline](classification-baseline/README.md)
- [Local sentiment evaluation](local-sentiment-evaluation/README.md)
- [Paired local sentiment evaluation](paired-sentiment-evaluation/README.md)
- [Amazon sentiment prompt evaluation](amazon-prompt-evaluation/README.md)
- [IMDb model and prompt evaluation](imdb-model-prompt-evaluation/README.md)
- [Incremental sentiment prompt instructions](prompt-instruction-ablation/README.md)
- [Final held-out sentiment validation](final-sentiment-validation/README.md)

## Series report

[Completed evaluation series and final prompt decision](FINAL_REPORT.md)

## New research series

- [Stage 1 — Independent-source binary sentiment validation](independent-source-validation/README.md)
- [Stage 2 — Neutrality, ambiguity and human-review deferral](uncertain-sentiment-evaluation/README.md)
- [Stage 3 — Sentiment stability under meaning-preserving edits](sentiment-stability/README.md)
- [Stage 4 — Local quality, latency and allocation tradeoffs](local-resource-tradeoffs/README.md)

[Four-stage series report and operational recommendation](NEW_SERIES_REPORT.md)
