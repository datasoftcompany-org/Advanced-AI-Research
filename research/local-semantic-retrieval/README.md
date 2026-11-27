# Local embeddings and semantic retrieval

## Question and fixed fixture

Can the installed Nomic embedding model retrieve an English support passage when a question uses different wording? This stage compares dense retrieval with a lexical BM25 baseline.

The fixture and protocol were committed before inference in `650683e`. It contains 16 short fictional support-policy documents and 32 answerable questions: one direct and one paraphrased question for each document. Each question has one authored relevant document ID. Documents and references were co-authored without independent annotation.

[Documents](documents.json) · [Questions](queries.json) · [Protocol](protocol.json) · [Input hashes](manifest.json) · [Rankings and timing](results.json)

## Implementation

Inference uses the already-installed `nomic-embed-text:v1.5`, whose digest is recorded in the result. No model was downloaded. We use `search_document: ` and `search_query: ` task prefixes as specified by the [Nomic model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5). Input is sent to the local [Ollama embedding endpoint](https://docs.ollama.com/api/embed) with truncation disabled and context 2048.

Every document is one passage, including title and text. Embeddings are validated for dimensions, finite values and nonzero norm, normalized to unit length and rounded to float32. Search computes dot products against the normalized query. Ties are resolved by document ID.

The baseline uses BM25 over lowercase alphanumeric tokens, with k1=1.5 and b=0.75, without stemming or stopword removal. It searches the same document titles and contents and uses the same tie rule. It re-tokenizes the tiny corpus per query in this implementation, so its measured latency is specific to this baseline code.

Two rounds rebuild the document index and encode each query independently. The sample and order remain fixed. Hit@1 and Hit@3 measure whether the relevant passage appears in the first one or three results; MRR is mean reciprocal rank.

## Results

Recorded setup: Python 3.13.4, arm64, Ollama 0.40.16. Every request returned a valid 768-dimensional embedding. Complete document rankings repeated between rounds.

| Retrieval method | Hit@1 | Hit@3 | MRR |
| --- | ---: | ---: | ---: |
| Nomic cosine retrieval | 29/32 (90.6%) | 32/32 (100%) | 0.953 |
| BM25 baseline | 23/32 (71.9%) | 26/32 (81.3%) | 0.775 |

Nomic retrieved all 16 direct questions correctly at rank one. For paraphrases it retrieved 13/16 correctly at rank one and 16/16 within the top three. The three misses were domestic returns versus international returns, domestic express delivery versus international delivery, and cancellation versus refund processing. In all three, the referenced passage ranked second.

### Latency and index size

| Measurement | Round 1 | Round 2 |
| --- | ---: | ---: |
| Document embedding batch time, including any load | 1.533 s | 0.231 s |
| Median query embedding request | 17.7 ms | 15.8 ms |
| Median dense similarity ranking | 2.41 ms | 2.31 ms |
| Median BM25 ranking | 0.63 ms | 0.58 ms |

The document vector matrix uses 49152 bytes (48 KiB) for 16×768 little-endian float32 values. The metadata file adds 945 bytes per index. These figures exclude source documents, query vectors, model weights and process memory.

Both binary indices, metadata and query vectors remain local at `/Users/m3/Documents/AI-Research-Outputs/semantic-retrieval-01`. The public repository contains the small authored texts, hashes and ranking results.

## Interpretation and next integration boundary

Embeddings improved paraphrase retrieval in this authored fixture, but confusion between similar policies remained. Retrieving the top three provides the reference here alongside distractors; a subsequent answer generator would still need to choose and cite the right passage.

The fixture contains only answerable questions. No score threshold for refusing unsupported questions was calibrated, and similarity values are not confidence probabilities. The 16 passages and co-authored questions provide a small diagnostic demonstration. This does not establish performance on long documents, mixed languages, larger corpora or naturally collected user questions.

## Reproduce locally

With the existing embedding model installed and Ollama running, choose a new output directory:

```sh
python3 -B research/local-semantic-retrieval/evaluate_retrieval.py --output /tmp/retrieval-new-run
python3 -B -m unittest discover -s research/local-semantic-retrieval -v
```

The runner verifies source hashes and model identity, rejects changed manifests, and writes each round's index and query vectors outside the repository. Compare saved ranks and metrics. Timings can vary with machine state.
