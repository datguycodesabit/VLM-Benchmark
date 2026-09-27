# Local IAM data layout

This folder contains local benchmark inputs and is intentionally excluded from
Git. Do not commit IAM images, transcriptions, or generated results.

The supplied archive was the IAM **word-image** release, not the form-image plus
XML release expected by the main `vlm-bench prepare --data data` command. It has
been sorted into:

```text
data/words/
  images/       44,564 word PNGs with a matching reference
  references/   44,564 UTF-8 references from IAM words.txt
  unlabeled/    70,756 word PNGs without a reference in the supplied text file
  metadata/     words.txt and words_new.txt
```

The original `data/archive.zip` is preserved. The extracted archive directory
was emptied after sorting. The importer now auto-detects this word layout when
you run `vlm-bench prepare --data data`; use `--limit 100 --seed 42` for a
reproducible 100-word subset. It skips unreadable/truncated word crops with a
warning. For form-level handwriting evaluation, add IAM offline form images
under `data/images`, matching XML under `data/xml`, and optional reference
overrides under `data/references`.
