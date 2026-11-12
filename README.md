# ICD-11 coder (MMS 2026-01)

Assigns up to three ICD-11 MMS codes to each clinical note. A hybrid retriever (BM25 plus BioLORD embeddings, fused with RRF) builds a ranked candidate pool. An optional LLM reranks the pool. With no API key, the retriever alone still answers every note.

## Run

From a clean clone, put the notes at `in/notes.jsonl`, then run:

```
docker compose build
docker compose run coder /in/notes.jsonl /out/predictions.jsonl
```

Predictions are written to `out/predictions.jsonl`. The image already contains the BioLORD-2023-M model and the catalogue embeddings, so `run` needs no network access beyond the model APIs.

API keys are optional. Set them in the shell, or copy `.env.example` to `.env`:

```
OPENAI_API_KEY=...        # optional
ANTHROPIC_API_KEY=...     # optional
HTTPS_PROXY=...           # optional; the OpenAI and Anthropic SDKs honour it
```

API spend is capped at US$0.85 (`BUDGET_USD` in `src/coder.py`). A call that would exceed the cap is refused, and those notes keep the retriever ranking. Predictions are written after every note, so a timeout does not lose finished work.

Local run without Docker (requires `uv`):

```
uv sync
uv run python -m src.coder --hybrid data/dev_notes.jsonl out/dev_predictions.jsonl
uv run python score.py --gold data/dev_notes.jsonl --pred out/dev_predictions.jsonl --mms data/mms-2026-01.json
```

Modes: default (LLM if a key is set, otherwise retriever only), `--hybrid` (retriever only, no API calls), `--api` (requires a key).

**Status:** Docker is not installed on the machine used for development, so the image build has not been run. The Python path was run end to end with the commands above.

## Scores (60 dev notes, official `score.py`)

| Setup | Score (of 75) | Top-1 | Top-3 |
|---|---|---|---|
| Retriever only, no key (`--hybrid`, measured) | 29.2 | 0.317 | 0.533 |
| Best recorded LLM run (gpt-4.1-mini rerank, term extraction, abbreviations; Decision 22) | 70.0 to 70.8 | not recorded | not recorded |

The dev notes were used to tune prompts and rules, so the LLM figures are optimistic. My estimate of the hidden score is 55 to 65 with an LLM and about 27 to 31 without a key. The estimate is a judgement, not a measurement: the dev set was used for tuning, so I expect the hidden score to be lower.

## Data added

- **Catalogue embeddings:** `data/.embeddings_cache/mms-2026-01_FremyCompany_BioLORD-2023-M.pkl`, computed from `data/mms-2026-01.json` with BioLORD-2023-M. The model is downloaded at image build time.
- **Abbreviation table and chapter priors:** written by hand in `src/coder.py` and `src/retrieval/priors.py`. The chapter multipliers come from the dev-set analysis in Decision 09.
- **Synthetic evaluation notes:** 104 notes in `data/synthetic/synthetic_eval.jsonl`, 30 of them non-English. They cover several note styles (typed consult, ambient transcript) and perturbations such as negation. I wrote the gold codes, so this set is a weaker signal than the dev set. It is gitignored and not shipped in the image.

## Evaluation beyond dev

- The synthetic set above, scored with the same `score.py`.
- Output checks on every run: missing predictions, invalid codes and duplicate ids are all reported by `score.py`. The dev runs had 0 of each.
- Pool-recall checks (is the gold code in the 30-candidate pool?) to separate retrieval failures from ranking failures.

## AI tooling error I caught

In Decision 21, the first failure table labelled three misses (dev-051, dev-056, dev-060) as reranking problems. The gold code was missing from the candidate pool for those notes, so they were retrieval problems. A pool-membership check showed this, and the table was corrected before the next change. See `DECISIONs.md`.

## Files

- `src/coder.py`: CLI, LLM reranking, budget and deadline handling.
- `src/retrieval/`: BM25, semantic, RRF fusion, term extraction, chapter priors.
- `Dockerfile`, `.dockerignore`, `docker-compose.yml`: packaging.
- `DECISIONs.md`: decision log with dates and commit hashes.
