# Local sentiment tool

Classify short English reviews using a locally installed Ollama model. This command-line release uses the direct prompt and generation settings selected in the [final research report](../../research/FINAL_REPORT.md).

## Browser interface

Start Ollama, then run from the repository root:

```sh
python3 tools/sentiment/serve.py
```

Open `http://127.0.0.1:8765`. The Arabic interface supports English review text, CSV upload with a configurable column, model selection, row-by-row progress and CSV download. It reuses the repository's colored glass card styling.

The server binds to the loopback address and runs one classification at a time. Uploaded content is processed in memory; the app writes no uploaded files or results to disk. Download saves the CSV through the browser. Requests are limited to 10 MiB including JSON encoding. The interface reports incomplete or invalid rows and only enables results after model identity verification. Closing the tab can interrupt the request; rerun the batch after the active request finishes.

Keep the terminal running while using the interface. Press Ctrl+C to stop it. Use `--port 8766` if the default port is occupied. The terminal prints the corresponding URL.

## Run

Use Python 3.13, start Ollama, and have either `llama3.2:3b` or `qwen2.5:7b` installed locally. No extra Python packages are required. From the repository root:

```sh
python3 tools/sentiment/classify.py --text "The food was excellent."
```

The default is Llama. Choose Qwen and export to a new CSV with:

```sh
python3 tools/sentiment/classify.py --text "I would happily return." --model qwen2.5:7b --output /tmp/sentiment-result.csv
```

For a UTF-8 CSV containing a `text` header:

```sh
python3 tools/sentiment/classify.py --csv /path/to/reviews.csv --output /path/to/results.csv
```

Use `--column review` if the review column has another name. Input may contain quoted commas, multiline reviews and a UTF-8 BOM. A header and at least one data row are required. Extra source columns are accepted and the output contains the selected text column and source row number.

Without `--output`, results are printed as JSON. CSV exports include `row`, `text`, `sentiment`, `status`, `model`, `model_digest`, and `ollama_version`. Potential spreadsheet formula cells are prefixed with an apostrophe in the exported CSV. JSON retains the original text.

## Status and errors

Successful rows have `ok` status and a positive or negative sentiment. Empty text rows have `empty` status. Responses outside the exact permitted labels have `invalid` status. Interrupted or unsuccessful requests have `failure` status. These rows have a blank sentiment and retain their source row numbers.

Exit code 0 means all rows succeeded. Exit code 2 means some rows were empty, invalid or failed; the completed results are still exported. Exit code 1 means the input, model service, or export failed. Correct the issue and rerun the input with a new output filename.

The tool verifies the model digest and service version before and after the batch. Existing output files are preserved. The CSV is assembled before creating the destination exclusively. A failed write removes the incomplete export. Output directories may be on the local disk or the shared exFAT disk.

## Scope and validation

This release processes short English reviews on the local machine. It submits text to Ollama at 127.0.0.1 and verifies that the selected model is local. Language detection and long-document chunking are not implemented. The 2048-token context and eight-token generation limit match the research settings. Review length and truncation can affect results.

Validation covered real requests with both models and a batch with positive, negative and empty rows. Regression checks cover malformed input, multiline CSV, invalid responses, interrupted requests, model identity changes and output preservation.

```sh
python3 -B -m unittest discover -s tools/sentiment -v
```
