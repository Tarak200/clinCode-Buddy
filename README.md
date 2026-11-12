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

## Scores (official `score.py`)

| Setup | Set (scale) | Score | Top-1 | Top-3 |
|---|---|---|---|---|
| Retriever only, no key (`--hybrid`) | 60 dev notes, core (of 75) | 29.2 | 0.317 | 0.533 |
| Best LLM run (gpt-4.1-mini rerank, term extraction, abbreviations; Decision 21, runs 2 and 3) | 60 dev notes, core (of 75) | 70.8 | 0.933 | 0.967 |
| LLM run 1 of the same setup (Decision 21) | 60 dev notes, core (of 75) | 70.0 | 0.917 | 0.967 |
| Retriever only, no key (`--hybrid`) | 104 synthetic notes, generalisation (of 25) | 16.6 | 0.615 | 0.760 |
| LLM rerank on the retriever | 104 synthetic notes, generalisation (of 25) | 23.2 | 0.913 | 0.952 |

Dev validation for the 70.8 run: 60 notes, 0 missing predictions, 0 invalid codes, 0 duplicate ids. Mean of the three LLM runs is 70.5 (top-1 0.928, top-3 0.967). The dev notes were used to tune prompts and rules, so the dev LLM figures are optimistic. The synthetic set was not used for tuning, but its gold codes were not checked by a clinician, so it is also optimistic.

My estimate of the hidden score (out of 100, a judgement and not a measurement) is about 40 to 46 with no key and about 73 to 88 with an LLM. It is built from two parts: core 25 to 29 (no key) or 55 to 65 (LLM), discounted from the dev scores for tuning; and generalisation 14 to 17 (no key) or 18 to 23 (LLM), from the synthetic set.

## Data added

- **Catalogue embeddings:** `data/.embeddings_cache/mms-2026-01_FremyCompany_BioLORD-2023-M.pkl`, computed from `data/mms-2026-01.json` with BioLORD-2023-M. The model is downloaded at image build time.
- **Abbreviation table and chapter priors:** written by hand in `src/coder.py` and `src/retrieval/priors.py`. The chapter multipliers come from the dev-set analysis in Decision 09.
- **Synthetic evaluation notes:** 104 notes in `data/synthetic/synthetic_eval.jsonl`, 34 of them non-English. They cover three note styles (typed consult, ambient transcript, short consult). Its gold codes were not checked by a clinician, so it is a weaker signal than the dev set. It is not used at run time.

## Evaluation beyond dev

- The synthetic set above, scored with the same `score.py`: retriever only 16.6 of 25, LLM rerank 23.2 of 25.
- Output checks on every run: missing predictions, invalid codes and duplicate ids are all reported by `score.py`. The dev and synthetic runs had 0 of each.
- Pool-recall checks (is the gold code in the 30-candidate pool?) to separate retrieval failures from ranking failures.

## AI tooling error I caught

In Decision 20, the first failure table labelled three misses (dev-051, dev-056, dev-060) as reranking problems. The gold code was missing from the candidate pool for those notes, so they were retrieval problems. A pool-membership check showed this, and the table was corrected before the next change. See `DECISIONS.md`.

## Files

- `src/coder.py`: CLI, LLM reranking, budget and deadline handling.
- `src/retrieval/`: BM25, semantic, RRF fusion, term extraction, chapter priors.
- `Dockerfile`, `.dockerignore`, `docker-compose.yml`: packaging.
- `DECISIONS.md`: decision log with dates and commit hashes.

Time spent on this submission: about 31 hours.
