# Local image description and spatial reasoning pilot

## Images, rights and transfer size

Four CC0 still-life images were selected from the Metropolitan Museum of Art collection as distributed on Wikimedia Commons. The source file pages explicitly identify the image files under CC0 1.0; source URLs, download URLs, file sizes, preprocessing and image hashes are in [sources.json](sources.json).

- [Georg Flegel, Still Life](https://commons.wikimedia.org/wiki/File:Still_Life_MET_DT1572.jpg)
- [Paul Cézanne, Still Life with Jar, Cup, and Apples](https://commons.wikimedia.org/wiki/File:Still_Life_with_Jar,_Cup,_and_Apples_MET_DP130322.jpg)
- [Still Life with Books](https://commons.wikimedia.org/wiki/File:Still_Life_with_Books_MET_DP139567.jpg)
- [Still Life with Strawberries](https://commons.wikimedia.org/wiki/File:Still_Life_with_Strawberries_MET_DT1577.jpg)

Successful image downloads totalled 107653 bytes (105.1 KiB). This counts the four thumbnail payloads, excluding source-page requests and failed HTTP attempts. Direct 512-wide thumbnail requests were rejected by the source server, so published 330-wide thumbnails were downloaded instead. Each was enlarged proportionally to fit a 512×512 canvas, centered with light-gray padding, stripped of metadata and saved as JPEG. No content was cropped. Final image files total 164965 bytes (161.1 KiB). These are 512×512 files derived from lower-resolution previews.

The file sizes and licenses concern images only. All inference used the two already-installed local models and no model weights were downloaded.

## Frozen experiment

The images were visually inspected before writing the references. The fixture contains six yes/no questions per image, three yes and three no, for 24 questions in total. It covers object presence and relative left/right, above/below and foreground/background relationships. Four negative controls ask about a red car, one in each image. References are author annotations without independent review.

Images, sources, questions and protocol were frozen in commit `fcda1e4` before inference. The questions use a strict completed lowercase yes/no response. Invalid and failed requests remain in the denominator. The balanced constant-answer baseline is 12/24.

Both models receive each question in a fresh local chat with only that image and prompt. Image filenames and reference answers are not included in model input. Each image also receives a free-description request. Models run sequentially and then in reversed order, with identical question order. Model identities and speech-independent vision capabilities are checked locally.

[Questions and references](cases.json) · [Protocol](protocol.json) · [Input hashes](manifest.json) · [Raw answers and descriptions](results.json)

## Results

Recorded environment: Python 3.13.4, arm64, Ollama 0.40.16. All 96 question requests returned valid yes/no responses and all 16 description requests completed. Question predictions and description text repeated exactly across rounds.

| Model | Questions correct /24 | Accuracy | Median question time, rounds 1 / 2 |
| --- | ---: | ---: | ---: |
| qwen2.5vl:7b | 24 | 100.0% | 142 / 132 ms |
| gemma3:4b | 16 | 66.7% | 2817 / 2833 ms |

Gemma's errors were concentrated in spatial questions. It reversed the pitcher/glass relation and plate height in the first image, cup/jar and apples/cup relations in the second, open-book side in the third, and fruit/asparagus and flower/fruit relations in the fourth. Both models rejected all four red-car controls.

The timing comparison applies to these installed packages, image encoders, prompts and this machine. Smaller parameter count did not translate to lower measured request latency here. It does not establish a general latency ranking.

## Qualitative description review

The free descriptions were inspected against the displayed 512×512 fixtures after inference. This is a qualitative author review, without an independently scored caption benchmark.

Qwen correctly named several prominent objects and relations, including the jar, cup, books, open-book location, red fruit and asparagus. In the tableware image it described the yellow cheese-like object as a loaf of bread, although the authored question calls it cheese. That identity is uncertain at thumbnail resolution. Its perfect closed-question score therefore does not establish that every descriptive detail is correct.

Gemma's tableware description introduced a second glass and called another item a turnip. Its final still-life description mentioned fish where the foreground visibly contains asparagus. Small fruit identities also differed between the captions and source titles. The models did not consistently follow the requested maximum of three sentences. The raw descriptions preserve these observations for review.

## Interpretation and limits

Qwen is the stronger candidate for the next local vision prototype on this fixture. Its result needs broader photographic scenes before a general recommendation. This pilot covers three paintings and one still-life photograph, with only four independent images and repeated negative controls. Captions, art knowledge, annotation ambiguity, low-resolution enlargement and padding can affect answers.

The experiment does not evaluate precise counting, OCR, bounding boxes or real-world depth. Left and right refer to the viewer's image coordinates. No context-truncation or caption-completeness measurement was added. The closed questions and free descriptions support different conclusions and should be reviewed together.

## Reproduce offline

The four small images are included with source and license records. With both existing models available and Ollama running, execute from the repository root:

```sh
python3 -B research/local-vision-spatial/evaluate_vision.py
python3 -B -m unittest discover -s research/local-vision-spatial -v
```

The runner checks image and protocol hashes, records model digests, and rejects a changed manifest. The images and inputs are sufficient for local reproduction. Timings can vary with machine state.
