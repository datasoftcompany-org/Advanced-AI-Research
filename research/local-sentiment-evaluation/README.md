# Local sentiment evaluation

## Research question

How does a local Llama 3.2 3B model compare with a majority-class baseline on short English reviews? We also examine a separate set of negation and mixed or implicit sentiment examples.

## Protocol and data

We fixed the protocol and diagnostics before inference in commit `66d80c7`. The evaluation uses the Yelp portion of [Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) by Kotzias et al. (2015), distributed by UCI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original corpus is fetched separately.

The tool numbers rows by their original line position. After case-folding and collapsing whitespace for duplicate detection, it retains the first occurrence of each group with consistent labels and drops groups with conflicting labels. It ranks rows within each class by SHA-256 of the fixed salt and row ID. The first 12 per class form the test sample, with all remaining rows forming the baseline training split. Training counts were 487 positive and 485 negative. The baseline therefore always predicts positive.

The model receives one review per request without its label. After trimming whitespace, only the exact labels `negative` and `positive` are accepted. Invalid answers and request failures are recorded separately and count as missed predictions in accuracy and Macro-F1. The eight authored diagnostics are evaluated separately from the UCI sample. [Protocol settings](protocol.json) · [Diagnostic cases](diagnostics.json)

## Reproduce

Use Python 3.13 and a running local Ollama instance with `llama3.2:3b` already installed. From the repository root, download the dataset archive and run the evaluator:

```sh
curl -fL 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip' -o /tmp/sentiment-331.zip
python3 -B research/local-sentiment-evaluation/evaluate.py --archive /tmp/sentiment-331.zip
```

Compare the output with [results.json](results.json), including the model digest, input hashes, and per-row predictions. The evaluator checks the archive digest before reading it and records the local model digest and service version. The recorded runs used Python 3.13.4 on arm64 and Ollama 0.40.16.

Run the regression checks without Ollama:

```sh
python3 -B -m unittest discover -s research/local-sentiment-evaluation -v
```

## Results

Two consecutive runs on the same machine produced byte-identical JSON. Both runs recorded zero invalid answers and zero request failures.

| Split | Majority accuracy | Majority Macro-F1 | Llama accuracy | Llama Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| UCI Yelp | 12/24 | 0.333 | 24/24 | 1.000 |
| Negation | 2/4 | 0.333 | 4/4 | 1.000 |
| Mixed or implicit | 2/4 | 0.333 | 3/4 | 0.733 |

The Yelp confusion matrix has 12 correct predictions in each class and no off-diagonal entries. The model's only diagnostic error was `a4`. It predicted negative for “We booked another table before we even finished dessert,” where the repeat booking was intended to signal approval. The JSON report includes all confusion matrices and predictions, indexed by source row or diagnostic ID.

The balanced 24-row sample covers one source domain. The diagnostic cases use author-assigned labels. The public corpus may have appeared in model training. The repeated run checks reproducibility on one setup, and the scores do not estimate performance on broader or unseen review populations.
