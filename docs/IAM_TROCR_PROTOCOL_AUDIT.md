# IAM / TrOCR evaluation audit — November 2, 2026

## Finding

The saved 15.55% CER / 59% WER run is an exploratory isolated-word benchmark,
not a reproduction of TrOCR's published IAM result. No inference was rerun for
this audit. Differences below are verified settings, not measured causal effects.

## IAM has multiple tasks and partitions

The original IAM dataset provides scanned forms, sentences, lines and words.
Source scans are 300 dpi; a tiny segmented word is not evidence of low-resolution
page scanning. Its documented large writer-independent line task contains 6,161
training, 900 first-validation, 940 second-validation and 1,861 test lines. Writers
are separated across those sets. Images and split IDs are separate downloads.

Source: [original IAM archive and protocol](https://zenodo.org/records/19003721).

## TrOCR's published experiment

The paper specifies the **Aachen partition**, citing Laia: 6,161 training lines,
966 validation lines and 2,915 test lines. This differs from the 1,861-line test
task above. It evaluates line images, uses a 384×384 input, beam size 10, and
case-sensitive CER. Table 4 reports 2.89% CER for TrOCR-Large; that number is not
WER. The corresponding table does not supply our model's WER comparison.

Source: [TrOCR paper, dataset/settings/metrics and Table 4](https://arxiv.org/html/2109.10282v5).

The official IAM evaluation command uses `--beam 10 --scoring cer2 --gen-subset
test --input-size 384 --preprocess DA2`. Microsoft also publishes an IAM data
archive linked from the repository. Its exact sample IDs and scorer behavior
should be pinned before claiming a faithful numerical reproduction.

Source: [Microsoft IAM evaluation instructions](https://github.com/microsoft/unilm/tree/master/trocr#evaluation-on-iam).

## Our actual saved experiment

Evidence: `runs/trocr-iam-test/20261102-002304-d0fc5bf8/manifest.json` and
`results.jsonl`.

| Item | Saved setting |
|---|---|
| Images | 100 selected IAM words, not the Aachen line test |
| Snapshot | `6f702c4ab24a3e9bbdc4c0517705caa00a1cfe44c28c28caa7bcc54b46629965` |
| Checkpoint | `microsoft/trocr-large-handwritten` |
| Revision | `e68501f437cd2587ae5d68ee457964cac824ddee` |
| Runtime | PyTorch MPS on this Mac |
| Beam count | 1 (greedy), not the paper's 10 |
| Output budget | 32 new tokens; zero saved truncation flags |
| Repetitions | 1 |
| Text prompt | Stored by the common engine, but not consumed by TrOCR |
| CER | 95 character edits / 611 reference characters = 15.5483% |
| WER | 59 word edits / 100 reference words = 59% |
| Exact-match rate | 58/100 |

The adapter uses the checkpoint's official Hugging Face processor, RGB pixels,
`eval()` and inference mode; generation is deterministic greedy decoding. The
cached processor configuration resizes to 384×384, normalizes with mean/std 0.5,
and uses bilinear interpolation (`resample=2`). Microsoft's original DA2 evaluation
transform uses bicubic resizing. Thus the basic input size/normalization agree,
while interpolation and framework paths are not identical.

Sources: cached `preprocessor_config.json` at the pinned checkpoint revision;
[Microsoft preprocessing code](https://github.com/microsoft/unilm/blob/master/trocr/data_aug.py).

Our metric collapses whitespace, retains case and punctuation, and aggregates
edit counts over reference lengths. Exact compatibility with the authors'
`cer2` implementation has not been established. Do not substitute their
scene-text `wpa` scorer: that deliberately lowercases and filters characters,
and is a different task/metric.

## Interpretation and next experiment

1. The line/word task and dataset partition mismatches prevent direct comparison
   with 2.89%. This does not establish that IAM images are generally poor.
2. Beam count is a confirmed, immediately testable inference difference.
   `examples/trocr-iam-word-beam10.toml` retains the same word snapshot and
   32-token budget, changing greedy decoding to ten beams. It is an ablation,
   not a paper reproduction.
3. Zero truncation flags mean the word run does not provide evidence that its
   32-token cap caused the gap. A line experiment needs a separately adequate
   output budget and explicit truncation checks.
4. A reproduction requires the Aachen test IDs/data used by the authors, the
   matching references and checkpoint, beam-10 decoding, matched preprocessing
   and verified metric normalization. The alternative official 1,861-line task
   must not be mislabeled as the paper's 2,915-line task.
5. Readability-selected subsets remain useful for the project's separate research
   question, but should retain that label and cannot reproduce published IAM scores.

The official public IAM archive now lists `lines.tar` (683.4 MB), `ascii.tar`
(7.8 MB) and task split IDs, so line data is available without relying on the
old FKI site's unreachable pages. No dataset archives were downloaded in this audit.
