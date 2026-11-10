# ICD-11 lexical baseline

This is the first local retrieval baseline, not the finished Docker submission.
It uses BM25 over the supplied MMS titles and does not call an API or require
additional packages. It does not yet interpret negation, history, or clinical
specificity, so its predictions should not be used for clinical decisions.

From the project root, run:

```powershell
python -m src.coder data\dev_notes.jsonl predictions\dev_predictions.jsonl
python score.py --gold data\dev_notes.jsonl --pred predictions\dev_predictions.jsonl --mms data\mms-2026-01.json --per-note predictions\dev_results.jsonl
```

The predictions file contains up to three valid codes
per ID and is written incrementally. If a note has no matching title terms, the
program warns on stderr and returns zero-score codes in deterministic code
order; these are not evidence-based matches. Input and output directories must
already exist.