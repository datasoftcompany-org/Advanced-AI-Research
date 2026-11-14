# Paired local sentiment evaluation

## Research question

Does Qwen 2.5 7B improve on Llama 3.2 3B on a larger held-out sample of English reviews? How do the models differ on diagnostic cases and warmed request latency?

## Protocol

The protocol and example IDs were committed before inference. The primary sample contains 60 positive and 60 negative Yelp sentences from the [UCI Sentiment Labelled Sentences dataset](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), Kotzias et al. (2015), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The evaluator reuses the deduplication and class-stratified hash ranking from [phase two](../local-sentiment-evaluation/README.md), with a new salt, and excludes the 24 prior test IDs from both splits. The remaining 427 positive and 425 negative labels give a positive majority baseline.

The separate diagnostic set has 12 negation cases and 12 mixed or implicit sentiment cases. Each group has six examples per label. The models see identical text, prompting, options, and example order, with one request per row. Each model gets one unscored warm-up, then runs sequentially and is unloaded before the next model. The second round reverses model order. Timing is wall-clock latency per completed request, including HTTP and inference, excluding warm-up and request failures. Completed invalid responses would still count toward latency.

[Protocol](protocol.json) · [Frozen sample](sample.json) · [Diagnostic cases](diagnostics.tsv) · [Model digests and per-row results](results.json)

## Reproduce

Use a running local Ollama instance with `llama3.2:3b` and `qwen2.5:7b` installed. The recorded setup used Python 3.13.4 on arm64 and Ollama 0.40.16.

```sh
curl -fL 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip' -o /tmp/sentiment-331.zip
python3 -B research/paired-sentiment-evaluation/compare.py --archive /tmp/sentiment-331.zip
```

Check the returned digests, sample IDs, predictions and confusion matrices against the published result. Request timings vary between runs. Run the split and alignment tests independently of Ollama with `python3 -B -m unittest discover -s research/paired-sentiment-evaluation -v`.

## Results

Each model returned the same predictions in both rounds, with zero invalid responses and zero request failures. The baseline was 50% accurate with Macro-F1 of 0.333 in all three groups.

| Group | Llama accuracy | Llama Macro-F1 | Qwen accuracy | Qwen Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Yelp | 116/120 | 0.967 | 117/120 | 0.975 |
| Negation | 12/12 | 1.000 | 11/12 | 0.916 |
| Mixed or implicit | 9/12 | 0.748 | 7/12 | 0.580 |

On Yelp, Qwen alone got two examples right and Llama alone got one right. Both models classified all 60 negative examples correctly. On diagnostics, Llama alone got three examples right. Qwen misclassified the positive negation case n01, the negative ironic portion-size remark m06, and the positive implicit approval of the cake in m11. The [per-row results](results.json) include shared errors and all disagreements.

Median warmed Yelp latency was 81.1 / 81.6 ms for Llama and 168.5 / 168.8 ms for Qwen across rounds one and two. On this setup, Qwen's median was about twice Llama's on Yelp.

The net Yelp difference was one example. The sample represents one source and the diagnostic labels were authored for this study. The public corpus may overlap model training, and the timings characterize this machine, model digests and request configuration.
