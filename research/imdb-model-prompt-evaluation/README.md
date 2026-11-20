# IMDb model and prompt evaluation

## Research question

Does the direct prompt's advantage on Amazon repeat on movie reviews with both Llama 3.2 3B and Qwen 2.5 7B? We compare the same two prompts from phase four and the same model digests used previously.

## Data and protocol

The protocol and sample were committed before inference in `2d59e63`. We use 120 IMDb reviews, balanced by label, from [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), Kotzias et al. (2015), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The full archive is downloaded separately and verified by SHA-256.

The source member contains 1000 labelled records over 1002 physical lines. A record ends at a tab followed by 0 or 1 at the end of a line. Continuation lines are preserved with newline separators. IDs use the terminal physical line number. Unterminated final records are rejected.

The shared reader normalizes case and whitespace for duplicate detection, keeps the first occurrence of consistently labelled duplicates, and excludes conflicting groups. Ranking uses SHA-256 of a fixed salt and row ID within each class. After the 60-per-class test sample, 439 negative and 438 positive records remain for determining the negative majority baseline. The sample manifest records every selected text hash and label.

We run four conditions sequentially and then reverse their entire order. Each condition gets one unscored warm-up and the model is unloaded afterward. HTTP and generation latency are measured per completed request. Timings exclude warm-up and failures. The settings retain a 2048-token context and an eight-token output limit from previous phases; prompt or review truncation was not measured. Exact labels after whitespace trimming are accepted.

[Protocol](protocol.json) · [Frozen sample](sample.json) · [Predictions, model digests and timings](results.json)

## Reproduce

The recorded setup used Python 3.13.4 on arm64 and Ollama 0.40.16. Have both local models installed and Ollama running, then run from the repository root.

```sh
curl -fL 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip' -o /tmp/sentiment-331.zip
python3 -B research/imdb-model-prompt-evaluation/evaluate_imdb.py --archive /tmp/sentiment-331.zip
```

Check the model digests and sample hashes before comparing per-row predictions and aggregate metrics. Latencies can vary between executions. Test multiline parsing with `python3 -B -m unittest discover -s research/imdb-model-prompt-evaluation -v`. The other evaluation directories retain their regression tests.

## Results

Each condition returned identical predictions in both rounds. All 960 scored requests completed with valid labels and zero request failures. The majority baseline scored 60/120 with Macro-F1 of 0.333.

| Model and prompt | Correct | Accuracy | Macro-F1 | Median latency, rounds 1 / 2 |
| --- | ---: | ---: | ---: | ---: |
| Llama, direct | 115/120 | 95.8% | 0.958 | 83.0 / 82.3 ms |
| Llama, overall | 98/120 | 81.7% | 0.810 | 82.3 / 82.7 ms |
| Qwen, direct | 117/120 | 97.5% | 0.975 | 170.7 / 172.6 ms |
| Qwen, overall | 113/120 | 94.2% | 0.942 | 171.0 / 170.5 ms |

Within Llama, the direct prompt alone correctly classified 17 examples; within Qwen it alone correctly classified four. The overall prompt had no exclusive correct examples for either model. Llama correctly classified all negative reviews under both prompts, but positive correct predictions fell from 55 to 38. Qwen retained 59 correct negative predictions and positive correct predictions fell from 58 to 54.

Under the direct prompt, Qwen alone classified three examples correctly and Llama alone classified one. Under the overall prompt, Qwen alone classified 17 correctly and Llama alone classified two. Every comparison ID and prediction is recorded in the JSON.

## Relation to earlier phases and limits

Phase three's direct-prompt Yelp sample gave Qwen 117/120 and Llama 116/120. Phase four's Amazon comparison gave Llama 115/120 with the direct prompt and 104/120 with the overall prompt. Here, the direct prompt advantage repeated for both tested models on IMDb.

The overall prompt was transferred verbatim from Amazon and asks about a product, service or purchase. Its domain wording can affect movie-review performance. Multiple instructions differ between prompts, so these results cannot identify the effect of negation or contrast instructions alone.

The IMDb sample adds a third domain within the same UCI collection. Model pretraining overlap is unknown. The samples are small, and timings describe this local configuration. Scores across phases use different review samples and support descriptive comparisons under the recorded conditions.
