# Local tool-calling pilot

## Scope and frozen tasks

This is stage three of the local capability research series. It tests choosing local tools, passing arguments, returning final answers, handling missing data, avoiding unnecessary tools and completing multiple operations.

The 14 tasks and deterministic intermediate outputs were frozen before inference, finalized in commit `410cdf3`. The fixture contains four arithmetic tasks, three stock lookups, an unknown SKU, two unit conversions, two no-tool tasks and two multi-operation tasks.

The only executable functions are arithmetic, distance conversion and a read-only dictionary stock lookup. The dispatcher checks argument keys, numeric values, units and tool names. It offers no shell commands, file writes or external services. Per task limits are six chat turns and eight calls.

[Tasks and expected outputs](tasks.json) · [Tool schemas and settings](protocol.json) · [Input hashes](manifest.json) · [Full tool traces](results.json)

## Method and primary scoring

Both installed models run the same tasks sequentially, then repeat with model order reversed. Qwen3 is explicitly run with thinking disabled. Qwen2.5 is run without a thinking parameter. Temperature is zero, seed 42, context 4096 and generation limit 512 tokens. Runtime is Python 3.13.4 on arm64 and Ollama 0.40.16.

The frozen primary criterion requires a completed JSON object with exactly one answer key, correct numeric or string type and value, the expected number of calls to each tool, no argument rejections, and expected intermediate tool results in the recorded order. Unknown-stock errors are valid tool results, distinct from rejected arguments or failed requests. The tasks and references are never sent together.

The primary criterion checks execution evidence and output compliance. It does not enforce waiting for one tool result before constructing every later dependent call. The dependency audit below explicitly examines that limitation.

## Results

Primary task outcomes, answers and call-name sequences repeated across rounds. Generated call identifiers and timings varied.

| Model | Primary passes /14 | Correct tool-result evidence /14 | Argument rejections | Median task time, rounds 1 / 2 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 8B, thinking disabled | 13 | 14 | 0 | 2.08 / 2.12 s |
| Qwen2.5 7B | 4 | 13 | 0 | 1.67 / 1.69 s |

Tool-result evidence counts exact expected intermediate results and tool multiplicities, irrespective of final-answer formatting. Each model respected both explicit no-tool tasks. Neither had a request failure. Qwen3 made 14 calls and Qwen2.5 made 15 per round.

Qwen3's primary failure was the zero-stock case: the tool returned numeric zero, but its final answer encoded zero as a string. Qwen2.5 returned several correct numeric results as standalone values instead of the required answer object. It also confused zero stock with an unavailable item. Its stock-minus task used a guessed quantity of 100 in a parallel arithmetic call and ultimately answered 90 instead of the reference 12.

### Post-run dependency audit

In the two-step arithmetic task, both models issued multiplication and the dependent addition together in a single assistant turn. The second call used a correct intermediate value of 72, but it was constructed before receiving the multiplication result. Qwen3 passed the primary criterion for that task, despite this failure to demonstrate grounded sequential dependency.

For stock-minus, Qwen3 waited for the stock result before calling arithmetic. Qwen2.5 issued a stock lookup and a dependent calculation together with an incorrect guessed value. Thus the primary score alone should not be used as a measure of grounded multi-step reliability. A stricter orchestration policy or a dedicated sequential test can address this separately.

## Interpretation

Qwen3 is the stronger candidate for a bounded local tool prototype under this prompt and output contract. The Qwen2.5 result shows why correct tool execution and final schema compliance need separate reporting. Formatting errors account for much of its low primary score.

The fixture is small and synthetic, with fixed tools and a single prompt. It does not evaluate general browsing, unrestricted computer use, adversarial inputs or autonomous long tasks. No conclusion about the models' broader reasoning ability follows from this run. The tested mode uses Qwen3's declared tool capability with thinking disabled. No models were downloaded.

## Reproduce locally

With both models installed and Ollama running, execute from the repository root:

```sh
python3 -B research/local-tool-calling/evaluate_tools.py
python3 -B -m unittest discover -s research/local-tool-calling -v
```

The runner verifies input hashes and model identities, validates calls before dispatch, and records assistant messages, function arguments, returned values, final outcomes and timing. It omits internal thinking fields from logs. Compare the semantic records rather than expecting identical call IDs or timings.
