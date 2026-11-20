# Local sentiment evaluation series

The seven-phase implementation and evaluation series is complete. It provides reproducible local sentiment experiments, exact protocol and model identification, per-example predictions, regression tests and a final held-out validation.

## Completed work

| Phase | Deliverable |
| --- | --- |
| 1 | Research layout, project template, GitHub workflow and public-page visual direction |
| 2 | [Local Llama evaluation](local-sentiment-evaluation/README.md), including diagnostic cases |
| 3 | [Paired model comparison](paired-sentiment-evaluation/README.md) on Yelp |
| 4 | [Prompt comparison](amazon-prompt-evaluation/README.md) on Amazon |
| 5 | [Two-model, two-prompt comparison](imdb-model-prompt-evaluation/README.md) on IMDb |
| 6 | [Incremental instruction comparison](prompt-instruction-ablation/README.md) across all three domains |
| 7 | [Final held-out validation](final-sentiment-validation/README.md) with paired confidence intervals |

## Final decision

Use the **direct prompt for both Llama 3.2 3B and Qwen 2.5 7B** in this documented local evaluation setting. The final sample scored 177/180 for Llama direct, 176/180 for Qwen direct, and 174/180 for Qwen with negation and contrast.

Qwen's selected-minus-direct accuracy difference was −1.11 percentage points, with a paired 95% stratified bootstrap interval of [−2.78, 0.00] percentage points. The additional instruction failed the adoption rule fixed before inference. All three conditions repeated their predictions in reversed rounds, with zero invalid answers and zero failures.

## Reproduction and scope

Start with the phase-seven README for the download and evaluation command. Its protocol fixes the exact prompts, inference settings and confidence method. Its sample and results identify the data and model digests and all predictions.

Evaluation uses short English reviews from the three parts of one public UCI collection. Prior exposure of these texts during model training is unknown. The recorded confidence interval describes sampling uncertainty conditional on the observed balanced sample. The series supports the documented prompt decision and measured local behavior. New datasets, languages, deployment requirements or model versions need their own validation.

The research work for this series ends with the phase-seven report and its integration into the main branch. Further product development or new experiments can be scoped separately.
