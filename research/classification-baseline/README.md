# Sentiment classification baseline

## Research question

Can a four-keyword rule classify more examples correctly than a majority-class baseline on a small synthetic English sentiment fixture?

## Setup

This example uses Python's standard library. It uses no external data or model weights. Five fixed labels determine the majority-class fallback. Six separate hand-written phrases form the evaluation fixture. The rule matches whole words and uses the majority label if the scores tie.

## Reproduction

From the repository root, run:

```sh
python3 research/classification-baseline/evaluate.py
```

## Results and limitations

Verified with Python 3.15.2. The majority baseline scored **3/6 (50.0%)** and the keyword rule scored **5/6 (83.3%)**. The rule misclassified “Not good” because it matches “good” and has no negation handling.

The hand-written examples illustrate how to report an evaluation and inspect an error. Scores on this fixture do not estimate performance on real text.
