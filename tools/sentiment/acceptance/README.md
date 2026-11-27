# Local v0.1 acceptance check

## Fixture and references

The authored fixture and reference labels were frozen before model inference, finalized in commit `55be91c`. [reviews.csv](reviews.csv) contains 60 short English reviews and three additional empty-input cases: empty text, spaces and a newline. [expected.csv](expected.csv) is kept separate and is not uploaded to the classifier.

The 60 reviews contain 30 positive and 30 negative examples. Each category has six of each label: plain sentiment, negation, mixed opinions, implicit sentiment and CSV formatting. Formatting cases include embedded commas, quotes, semicolons and multiline text. The examples and labels were authored for this project. They have not undergone independent annotation.

## Browser acceptance procedure

1. Start Ollama and run `python3 tools/sentiment/serve.py` from the repository root.
2. Open the printed local URL, select CSV and upload `tools/sentiment/acceptance/reviews.csv`. Keep the column name `text`.
3. Run Llama and then Qwen, downloading a separate CSV for each. Verify that progress reaches 63, all 63 source rows appear and the last three have empty status.
4. Upload a CSV with a missing text column, verify the visible error and hidden stale results, then switch to text mode and classify a valid review to verify recovery.
5. Compare each downloaded export to the references:

```sh
python3 tools/sentiment/acceptance/check_results.py /path/to/downloaded-results.csv
```

The checker verifies row count, source text, row numbering, model identity consistency and reference labels. It records input and reference file hashes. A classification mismatch appears in the error list.

## Recorded results

Chrome browser testing used the actual local interface on port 8766 with Python 3.13.4, arm64 and Ollama 0.40.16. The [result record](results.json) contains all exported rows, model digests, category scores and browser observations.

| Check | Llama 3.2 3B | Qwen 2.5 7B |
| --- | ---: | ---: |
| Plain sentiment | 11/12 | 12/12 |
| Negation | 11/12 | 11/12 |
| Mixed opinions | 12/12 | 12/12 |
| Implicit sentiment | 10/12 | 11/12 |
| CSV formatting | 12/12 | 12/12 |
| Correct review classifications | 56/60 (93.3%) | 58/60 (96.7%) |
| Empty cases correctly preserved | 3/3 | 3/3 |

Both runs displayed progress and produced downloadable exports with all source text and row order preserved. No request failures, invalid model replies or browser JavaScript errors were recorded. The page fit a 320-pixel viewport without document overflow. The missing-column error and subsequent successful text classification verified recovery.

## Error review and release decision

Both models labelled row 17, “The film did not waste a minute of my time,” negative, although the authored reference is positive. Both also labelled row 38, asking for the recipe before leaving a restaurant, negative, against the intended positive interpretation. Llama additionally misclassified the positive documentary example at row 4 and the repeat-purchase example at row 37.

The workflow checks passed. These classification errors remain in the evidence, and this fixture was not used to change the prompt or model. The tool is ready for a local v0.1 release with its existing English-review scope and explicit row statuses. Scores on this small authored fixture characterize these examples and cannot estimate performance on future user reviews.

## Scope of this release

Use the [tool instructions](../README.md) for browser and command-line startup, CSV input, model selection, result export and error handling. No application changes were required by this acceptance run. The nine existing tool and HTTP regression tests also passed.
