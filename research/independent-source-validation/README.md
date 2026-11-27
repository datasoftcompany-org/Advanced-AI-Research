# Independent-source binary sentiment validation

## Question and scope

How does the approved direct prompt transfer from the UCI review collection to a separately curated source? This first stage of the new research series evaluates English social posts from TweetEval's sentiment task. The change introduces a source and domain shift. The prompt retains the word “review” from the earlier protocol.

The source is [Cardiff NLP TweetEval](https://github.com/cardiffnlp/tweeteval), Barbieri, Camacho-Collados, Espinosa-Anke and Neves (2020), using its sentiment subset originating in [SemEval-2017 Task 4](https://aclanthology.org/S17-2088/), Rosenthal, Farra and Nakov (2017).

## License and provenance

The [dataset card](https://huggingface.co/datasets/cardiffnlp/tweet_eval#licensing-information) identifies the sentiment task's license as Creative Commons Attribution 3.0 Unported, and also references Twitter terms. This license statement applies to the sentiment task. Our repository stores sample hashes, labels and predictions; source post texts are downloaded separately from the official repository.

The [protocol](protocol.json) fixes source revision `4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66`, four file hashes, sampling salt, sample size and models. The protocol and [sample](sample.json) were committed before inference in `dcefdb6`.

## Sample and evaluation

The source test split has 12284 posts: 3972 negative, 5937 neutral and 2375 positive. We exclude neutral labels before selection. Case-folding and collapsed whitespace identify duplicates; consistently labelled groups retain their first record and conflicting groups are dropped. This leaves 6343 eligible binary records. SHA-256 ranking of a fixed salt and original one-based line ID selects 80 examples per label.

The majority baseline is positive, determined from the original training labels after excluding neutral: 17849 positive and 7093 negative. It scores 80/160 (50%) with Macro-F1 of 0.333 on the balanced sample.

We reuse the direct prompt, seed, context limit, output limit, strict label handling, metrics and model checks from the earlier evaluator. We run two sequential rounds in reversed model order with unscored warm-ups. Invalid answers count as missed classifications. The recorded setup is Python 3.13.4 on arm64, Ollama 0.40.16. Model digests, prediction IDs, confusion matrices and request timings are in [results.json](results.json).

## Results

Both models produced identical predictions in both rounds. No request failures occurred.

| Condition | Correct /160 | Accuracy | Macro-F1 | Invalid per round | Median latency, rounds 1 / 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama direct | 137 | 85.6% | 0.854 | 0 | 96.0 / 95.9 ms |
| Qwen direct | 140 | 87.5% | 0.876 | 1 | 178.1 / 178.8 ms |

Llama correctly classified 79/80 negatives and 58/80 positives. Qwen correctly classified 78/80 negatives and 62/80 positives, predicting negative for 17 positives and producing one unaccepted response on a positive example. The shared evaluator retains the invalid status and example ID, but does not retain the raw unaccepted response.

For descriptive context, the final UCI validation gave Llama 177/180 (98.3%) and Qwen 176/180 (97.8%) under the direct prompt. The independent-source scores are lower by 12.7 and 10.3 percentage points respectively. Samples, domains and annotation practices differ, so this comparison cannot isolate which change caused the difference.

## Interpretation and limits

The earlier review results do not establish equally high performance on social text. Positive posts account for most errors here. This stage provides evidence for testing domain suitability before extending the local review tool's scope.

This is a balanced binary subset of the original three-class task. Neutral detection remains unevaluated. Public data may overlap model pretraining even though its collection is separate from UCI. Two rounds measure stability on one setup; they do not increase the independent sample size. Prompt wording, context truncation, seed choice and source annotation uncertainty can affect findings. The existing local tool retains its documented short-English-review scope.

## Reproduce

Download files from the pinned official revision to a directory outside the repository:

```sh
mkdir -p /tmp/tweeteval-sentiment
for name in test_text.txt test_labels.txt train_labels.txt mapping.txt; do
  curl --fail --location --connect-timeout 20 --max-time 120 "https://raw.githubusercontent.com/cardiffnlp/tweeteval/4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66/datasets/sentiment/$name" -o "/tmp/tweeteval-sentiment/$name"
done
python3 -B research/independent-source-validation/evaluate_source.py --data /tmp/tweeteval-sentiment
```

Have both local models installed and Ollama running. Compare file hashes, model digests, sample IDs and predictions with the committed record. The runner rejects changed source files or a different sample manifest. Timings can vary.

```sh
python3 -B -m unittest discover -s research/independent-source-validation -v
```
