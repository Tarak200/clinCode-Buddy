## Decision 01: Fixing the basic plan to use for this problem

- **Date/time:** 10-Nov-2026, 14:17 IST
- **Commit:** None
- **Decision:** 
    The basic plan has been decided to start with the simple methodologies to complex based on the results and metrics
- **Reasoning:** As the task can be done in many ways, starting with simple methodology seems fair to understand where improvements are needed

## Decision 02: Implement the BM25 retriever baseline

- **Date/time:** 10-Nov-2026, 15:32 IST
- **Commit:** `TBD`
- **Decision:** Start with a dependency-free local BM25 index over the supplied MMS titles. I matched the words in description to the title in MMS data to return the three highest-ranked valid codes. There are no missing predictions but the matches with gold are worst. so, ranking is bad with the exact match.
- **Alternatives considered:** semantic embeddings, and API-based ranking.
- **Evidence:** The supplied catalogue provides valid codes and titles, making a lexical baseline possible without an API.
- **Results:**
  - Development notes: 60 (slice A)
  - Top-1 accuracy: 0.083
  - Top-3 accuracy: 0.150
  - Score: 7.9 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0

## Decision 03: Replace BM25 with semantic matching using embeddings

- **Date/time:** 10-Nov-2026, 15:45 IST
- **Commit:** `TBD`
- **Decision:** Implement semantic retrieval using SentenceTransformer embeddings (all-MiniLM-L6-v2). Replace keyword matching with cosine similarity over dense embeddings.
- **Reasoning:** 
  - BM25 fails because clinical terminology is highly varied. Notes may say "pneumonia" while ICD titles say "community-acquired lower respiratory tract infection"
  - Semantic embeddings capture medical concepts even when vocabulary differs
  - all-MiniLM-L6-v2 is lightweight (22MB), fast (~50ms/note), and validates well on medical similarity tasks
  - Embeddings can be precomputed and cached, making inference fast for the 60-minute constraint
- **Alternatives considered:** 
  - API-based ranking (Claude/GPT) - expensive and limited to ~$1 budget
  - Fine-tuning on dev notes - not enough data (60 notes)
- **Results** 
  - Slice A (n=60)
  - Top-1 accuracy: 0.15
  - Top-3 accuracy: 0.25
  - Points: 13.7 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0
  - **Score: 13.7 / 75**

- **Trade-offs:** 
  - Adds dependency on sentence-transformers and torch
  - Requires 76MB for embeddings cache (well within limits)
  - No fine-tuning for medical domain (acceptable for baseline)

## Decision 04: storing the embeddings of mms-2026.json 

- **Date/time:** 10-Nov-2026, 17:00 IST
- **Commit:** `done`
- **Decision:** To create the embeddings of mms-2026.json file
- **Reasoning:**
  - not to take more time when doing it afterwards.

## Decision 05: Implement hybrid retriever (semantic + lexical)

- **Date/time:** 10-Nov-2026, 21:40 IST
- **Commit:** `Done`
- **Decision:** Create hybrid index combining semantic (70%) and BM25 (30%) scoring. Each approach has complementary strengths.
- **Reasoning:**
  - Semantic: Captures meaning, handles synonyms and abbreviations
  - Lexical (BM25): Catches exact keyword matches, handles acronyms better
  - Hybrid (70/30): Leverages both, with semantic as primary signal
  - Example: "TB" (acronym) ranks higher with BM25, but semantic catches "tuberculosis" concept
- **Expected improvement:** Additional 5-10% over pure semantic
- **Implementation:** 
  - Get top-50 from each retriever
  - Normalize scores to [0, 1]
  - Compute weighted combination: 0.7 × semantic + 0.3 × lexical
  - Re-rank and return top-3

-  **Results**
  - Slice A (n=60)
  - Top-1 accuracy: 0.167
  - Top-3 accuracy: 0.250
  - Points: 14.6 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0
  - **Score: 14.6 / 75**

- **Rationale** The improvement is only marginal and the results are still bad, have to think about the data preprocessing and other. 

## Decision 06: Diagnosis of the explored methods

- **Date/time:** 10-Nov-2026, 21:42 IST
- **Commit:** `TBD`
- **Decision:** About 35% of notes never reach the top-50 (retrieval), and another ~37% are in the top-50 but not the top-3 (ranking). Neither a better ranker nor a better retriever alone gets past ~65%.
- **Reasoning:**
- The match between the title and description is not enough. Both the retriever and ranker are not working much. so, have to work on both mechanisms.

## Decision 07: RRF, assessment extraction, chapter prior, family metric

- **Date/time:** 10-Nov-2026, IST
- **Commit:** `TBD`
- **Decision:** Replace the 0.7/0.3 score mix in `HybridIndex` with reciprocal rank fusion (k=60) and add three no-key changes:
  - **RRF** (`src/retrieval/fusion.py`): fuses one ranked list per (query, retriever). Zero-score BM25 padding is dropped before fusion, since it is alphabetical filler and not evidence.
  - **Assessment extraction** (`src/retrieval/extract.py`): `A:`/`Dx`/`Imp:`/`Assessment:` for typed and short notes, the doctor's diagnostic sentence for transcripts, negated clauses ("malaria excluded") removed. The assessment is an *extra* query (weight 2); the full note is always kept as a fallback. Found in 58/60 dev notes.
  - **Chapter prior** (`src/retrieval/priors.py`): soft multipliers (0.85-0.95) on chapters 21-26, applied only if a disease-chapter code is in the fused top-10. It is never a filter, because gold can be in ch. 24 (`QA42.0` in dev) and the hidden set covers other chapters.
- **Reasoning:** Cosine scores (mapped from [-1,1]) and max-normalised BM25 are not on comparable scales; rank fusion needs no calibration. Most of the note is noise around a one-line diagnosis, and symptom/injury codes filled the top-3.
- **Results (60 dev notes, hybrid, official `score.py`):**

| Variant | R@1 | R@3 | R@10 | R@50 | Retrieval fails | Top-3 slots in ch. 21-26 |
|---|---|---|---|---|---|---|
| Old score-mix hybrid | .133 | .283 | .533 | .650 | 21 | n/a |
| RRF only | .217 | .350 | .500 | .750 | 15 | .372 |
| + assessment | .250 | .433 | .567 | .800 | 12 | .222 |
| + prior only (no assessment) | .233 | .383 | .567 | .750 | 15 | .239 |
| **RRF + assessment + prior (default)** | **.267** | **.483** | **.600** | **.800** | **12** | **.144** |

  - Score **25.4 / 75** (top-1 .267, top-3 .483), 0 missing / invalid / duplicate.
  - Sensitivity (assessment weight 1-3, prior strength 0-1.5x, RRF k 20-100): score stays within 23-26, i.e. within +-1 note, so defaults were left at the standard values rather than tuned to 60 notes.
  - Truncation question answered: only 5/60 notes exceed 256 word-pieces, so truncation is not a main cause.
- **Limits:** dev has no injury/symptom gold, so the prior's cost on those chapters is untested (hence the mild, guarded design). 12 notes still never reach the top-50 (vocabulary: next is the LLM extraction).

- **Alternatives considered:** 
  - API-based ranking (Claude/GPT) - expensive and limited to ~$1 budget

  **Conclusion:**
  We can use the RRF + priors + assessment combination when we don't have any API available as a fallback. Still it has the 12 true retrieval failures and remaining problem is mostly ranking.

## Decision 08: Usage of medical aware embedding model 

- **Date/time:** 11-Nov-2026, 16:30 IST
- **Commit:** `TBD`
- **Decision:** Instead of using the `all-MiniLM-L6-v2` which don't have any medical knowledge, I choose to use the `SapBERT` model which has medical context. Intially used the `NeuML_biomedbert` model as it is ultra-light and fast but it is not able to capture the semantics of the diagnosis. so, `SapBERT` is the best option.
- **Reasoning:** For `all-MiniLM-L6-v2` even the symptom is refering to a disease, it can't get the context while the `SapBERT` can.

- **Diagnosis Results:**

- **Exact-code recall**

| Retriever | Cohort (n) | R@1 | R@3 | R@10 | R@30 | R@50 |
|---|---:|---:|---:|---:|---:|---:|
| Lexical | All (60) | .083 | .150 | .300 | .450 | .517 |
| Lexical | Ambient transcript (15) | .000 | .133 | .467 | .667 | .733 |
| Lexical | Short consult (16) | .062 | .125 | .125 | .250 | .438 |
| Lexical | Typed consult (29) | .138 | .172 | .310 | .448 | .448 |
| Semantic | All (60) | .017 | .017 | .133 | .217 | .233 |
| Semantic | Ambient transcript (15) | .000 | .000 | .133 | .333 | .333 |
| Semantic | Short consult (16) | .000 | .000 | .188 | .250 | .250 |
| Semantic | Typed consult (29) | .034 | .034 | .103 | .138 | .172 |
| Hybrid | All (60) | .333 | .550 | .783 | .967 | 1.000 |
| Hybrid | Ambient transcript (15) | .400 | .800 | .800 | 1.000 | 1.000 |
| Hybrid | Short consult (16) | .375 | .500 | .750 | .875 | 1.000 |
| Hybrid | Typed consult (29) | .276 | .448 | .793 | 1.000 | 1.000 |

- **Family-level recall**

| Retriever | Cohort (n) | R@1 | R@3 | R@10 | R@30 | R@50 |
|---|---:|---:|---:|---:|---:|---:|
| Lexical | All (60) | .100 | .233 | .450 | .567 | .650 |
| Lexical | Ambient transcript (15) | .000 | .267 | .600 | .733 | .800 |
| Lexical | Short consult (16) | .062 | .125 | .250 | .312 | .500 |
| Lexical | Typed consult (29) | .172 | .276 | .483 | .621 | .655 |
| Semantic | All (60) | .117 | .167 | .283 | .400 | .433 |
| Semantic | Ambient transcript (15) | .200 | .200 | .333 | .400 | .400 |
| Semantic | Short consult (16) | .000 | .062 | .250 | .438 | .438 |
| Semantic | Typed consult (29) | .138 | .207 | .276 | .379 | .448 |
| Hybrid | All (60) | .617 | .767 | .867 | .967 | 1.000 |
| Hybrid | Ambient transcript (15) | .667 | .867 | .867 | 1.000 | 1.000 |
| Hybrid | Short consult (16) | .562 | .688 | .812 | .875 | 1.000 |
| Hybrid | Typed consult (29) | .621 | .759 | .897 | 1.000 | 1.000 |

- **Diagnostics:** Lexical had 29 retrieval failures (gold absent from top-50) and 22 ranking failures (gold in top-50 but not top-3); semantic had 46 and 13, respectively; hybrid had 0 and 27. Family-only misses (correct disease family in top-3, wrong specificity) were 5, 9, and 13, respectively. Symptom/injury/context chapters (21-26) accounted for 99/180 lexical, 83/180 semantic, and 27/180 hybrid top-3 slots.

- **Results (60 dev notes, hybrid retriever with SapBERT, `predictions\hybrid_only_sapbert_embedding_model.jsonl`, official `score.py`):**
  - Slice A (n=60)
  - Top-1 accuracy: 0.333
  - Top-3 accuracy: 0.550
  - Points: 30.4 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0
  - **Score: 30.4 / 75**

- **Observation** 
The medical aware embedding model is working much better with 0 retrieval failures for R@50 and 96.7% retrieval success at R@30 and the backdrop is only for short type inputs. so, the main issue here is the reranking them. LLM can help here provided. Symptom/injury/context chapters (21-26) accounted for 99/180 lexical, 83/180 semantic, and 27/180 hybrid top-3 slots. so, increasing penalty for the symptom chapters can help to give less weightage to these in top-3.

- **Truncation:** 0/60 notes exceeded 512 tokens; mean fraction of each note seen by the encoder was 1.00.
- **Conclusion:** Hybrid retrieval is the strongest of the three on this dev set: it retrieves a gold code into the top-50 for every note and reaches .550 exact-code R@3 / .767 family R@3. However, 27 notes with a gold code in the top-50 still miss the exact-code top-3, and 13 top-3 results identify the right disease family at the wrong specificity. Further work should focus on ranking and code specificity, not note truncation or retrieval coverage.

- **Alternatives considered:** 
  - changing factors in priors may help by giving very less weightage to chap 21-26
  - API-based ranking (Claude/GPT) - expensive and limited to ~$1 budget
  - First retrieving 50 and them 30 from the mms dataset to check how llm is helping

## Decision 09: Optimizing penalty for chap 21-26 in priors

- **Date/time:** 11-Nov-2026, 22:30 IST
- **Commit:** `Done`
- **Decision:** optimize the default chapter factors so that top-3 contains specific disease codes instead of the symptom codes

- **Results (60 dev notes, hybrid retriever with SapBERT, `predictions\hybrid_only_sapbert_embedding_model.jsonl`, official `score.py`):**
  - Slice A (n=60)
  - Top-1 accuracy: 0.350
  - Top-3 accuracy: 0.567
  - Points: 31.7 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0
  - **Score: 31.7 / 75**

- **Alternatives considered:** 
  - API-based ranking (Claude/GPT) - expensive and limited to ~$1 budget
  - First retrieving 50 and them 30 from the mms dataset to check how llm is helping

## Decision 11: LLM for diagnosis understanding to combat the ranking failures

- **Date/time:** 11-Nov-2026, 23:00 IST
- **Commit:** `TBD`
- **Decision:** To use LLM (Qwen/Qwen2.5-1.5B-Instruct) which has medical knowledge for the reranking of the codes within 30 candidates as R@30 is close to 1 for all types of inputs styles.

- **Reasoning:**
  - For reranking needs larger context understanding, specifically medical data. so, LLM which was pretrained on the medical data can help here.

-  **Results**
  - The inference time is too high as they are running on GPUs. 

- **Alternative Methods**
  - Use GGUF formatted model instead of FP16 which will occupy less memory.

## Decision 12: One LLM call per note for the local model

- **Date/time:** 11-Nov-2026, 23:35 IST
- **Commit:** `TBD`
- **Decision:** `llm_codes` in `src/coder.py` makes one rerank call per note instead of extract plus rerank. The assessment from `extract_assessment` is sent as `ASSESSMENT:` with a 1500-character note excerpt, instead of the full 8000-character note.
- **Reasoning:**
  - On a 4-core CPU with no GPU, prefill of about 2,000 note tokens was repeated for each of the two calls per note, and that dominated latency.
  - The extract call's diagnoses only fed extra retrieval queries. `HybridIndex.search` already adds the extracted assessment as a weighted query, so those queries were redundant.
  - The extract step could truncate its JSON at `LOCAL_MAX_NEW_TOKENS = 60`, which triggered a retry. The rerank output is only about 30 tokens, so that truncation no longer applies.
- **Trade-offs:**
  - The change applies to API providers too, so they lose the LLM-written diagnosis list. This is not yet measured on the dev set.
  - Timing is not yet measured for this change. The score is measured below.

  **Results (60 dev notes, single rerank call per note, official `score.py`):**
  - Slice A (n=60)
  - Top-1 accuracy: 0.233
  - Top-3 accuracy: 0.450
  - Points: 22.9 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0
  - **Score: 22.9 / 75**

  **Observation:** This is 8.8 points below Decision 09 (31.7 / 75, top-3 .567). The single-call change lowers ranking accuracy on dev, so it is a regression on score. Whether the latency gain justifies it is still open, because latency was not recorded for this run. It took ~32 min to run for 60 notes, so for 300 notes, it is going to take 160min which is more than our time. This can be trade-off by increasing threads. but still it is going to tak e ~80min for this task, so this is a backup. 

## Decision 13: Diagnosing the LLM rerank on top of the hybrid retriever

- **Date/time:** 12-Nov-2026, 02:00 IST
- **Commit:** `TBD`
- **Decision:** `misc/diagnose.py` gets `--llm-run`, which scores an LLM rerank run against the same 30-code hybrid pool that `src/coder.py` reranks. Each note is classified as kept, rescued, harmed, pool-miss or outside-pool. The earlier diagnostic only measured retriever recall and never read the LLM's picks.
- **Reasoning:**
  - Decision 12's LLM score (22.9 / 75) is below the hybrid retriever alone (31.7 / 75), but the score does not say whether the LLM or the retriever is at fault. The split by note shows which.
  - The pool and top-3 are rebuilt with `CANDIDATES` and `MAX_NOTE_CHARS` imported from `src/coder.py`, so the diagnostic cannot drift from the coder.

- **Command:** `uv run --offline python -m misc.diagnose --hybrid --llm-run predictions\llm_run.jsonl` (60 dev notes, SapBERT, `predictions\llm_run.jsonl`)

- **Hybrid retriever recall (exact, all 60 notes):**

  | R@1 | R@3 | R@10 | R@30 | R@50 |
  |---|---|---|---|---|
  | 0.350 | 0.567 | 0.783 | 0.967 | 1.000|

  - Family recall: R@1 0.633, R@3 0.783, R@10 0.867, R@30 0.967, R@50 1.000
  - Retrieval failures (no gold in top-50): 0
  - Ranking failures (gold in top-50, not top-3): 26
  - Family-only misses (right disease in top-3, wrong specificity): 13
  - Symptom/injury/context slots (chapters 21-26) in top-3: 16/180 = 0.089

- **LLM rerank on top of the hybrid pool:**

  | group | n | pool R@30 | retr@1 | LLM@1 | retr@3 | LLM@3 | retr fam@3 | LLM fam@3 | rescued | harmed | pool-miss | outside pool |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | all | 60 | 58 | 21 | 14 | 34 | 27 | 47 | 41 | 5 | 12 | 19 | 2 |
  | ambient_transcript | 15 | 15 | 6 | 6 | 12 | 9 | 13 | 11 | 0 | 3 | 3 | 0 |
  | short_consult | 16 | 14 | 7 | 2 | 9 | 4 | 12 | 9 | 1 | 6 | 4 | 2 |
  | typed_consult | 29 | 29 | 8 | 6 | 13 | 14 | 22 | 21 | 4 | 3 | 12 | 0 |

  - Net top-3 change from the LLM: **-7 notes** (rescued 5, harmed 12).
  - The LLM missed gold that was in the pool for 19 notes.
  - The LLM's top-1 and top-3 match `score.py` (0.233 and 0.450), so the diagnostic agrees with the official scorer.

- **Observation:** The LLM is the bottleneck, not retrieval. Gold is in the pool for 58/60 notes, but the LLM turns 12 correct top-3 lists into misses and rescues only 5. Harm is concentrated in `short_consult` (6 harmed, 1 rescued).

- **Alternatives considered:**
  - Keep the retriever's top-1 unless the LLM is confident.
  - Use a larger model. This is not measured yet. (openAI or Anthropic API)

## Decision 14: OpenAI gpt-4o-mini rerank on top of the hybrid retriever

- **Date/time:** 12-Nov-2026, 10:31 IST
- **Commit:** `TBD`
- **Decision:** Use the OpenAI API with `gpt-4o-mini` as the reranker on top of the hybrid retriever (SapBERT embeddings with RRF fusion, assessment extraction and chapter prior). The LLM chooses the final codes from the hybrid candidate pool, and the retriever supplies the candidates. The run uses `--api` with `OPENAI_API_KEY` set, instead of the local Qwen model from Decisions 11 to 13.
- **Reasoning:**
  - The local Qwen rerank was slow on CPU (about 32 min for 60 notes) and lowered the score (Decisions 12 and 13). Its misses were mainly reranking errors, since gold was in the pool for 58/60 notes.
  - A hosted, larger model should handle the ranking step better than a 1.5B local model, at a small per-note API cost.

- **Results (60 dev notes, hybrid retriever + `gpt-4o-mini` rerank, official `score.py`):**
  - Slice A (n=60)
  - Top-1 accuracy: 0.567
  - Top-3 accuracy: 0.850
  - Points: 49.6 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0
  - **Score: 49.6 / 75**

- **Comparison:**

| Run | Top-1 | Top-3 | Score |
|---|---|---|---|
| Hybrid retriever only (Decision 09) | .350 | .567 | 31.7 / 75 |
| Local Qwen2.5-1.5B rerank (Decision 12) | .233 | .450 | 22.9 / 75 |
| **Hybrid + gpt-4o-mini rerank (this run)** | **.567** | **.850** | **49.6 / 75** |

- **Observation:** The score rises by 17.9 points over the retriever alone and 26.7 points over the local LLM. Top-3 rises from .567 to .850, so the ranking failures that were the main problem in Decision 08 and Decision 13 are largely fixed by a stronger reranker. Retrieval is unchanged, so the pool ceiling still applies (hybrid R@30 .967 in Decision 13). Notes with the right disease family but the wrong specificity remain a likely source of the remaining misses. It had changed 95% of the top-3 output codes.

- **Limits:**
  - Measured on the 60 dev notes only. The hidden set may cover chapters and styles not in dev, so this is not a generalisation claim.
  - Per-note diagnostic (`misc\diagnose.py --llm-run`) has not been run for this predictions file, so the rescued/harmed split is not yet known.
  - Latency and API cost for this run are not recorded here.

- **Alternatives considered:**
  - OpenAI API (`gpt4.1-mini`) as the option
  - Anthropic API (`claude-haiku-4-5` is the default fallback in `src\coder.py`). Not measured.

## Decision 15: Use OpenAI gpt-4.1-mini for reranking

- **Date/time:** 12-Nov-2026, 11:11 IST
- **Commit:** `TBD`
- **Decision:** Use OpenAI's `gpt-4.1-mini` as the LLM reranker on top of the hybrid retriever.
- **Results (60 dev notes, official `score.py`):**
  - Slice A (n=60)
  - Top-1 accuracy: 0.783
  - Top-3 accuracy: 0.850
  - Points: 60.4 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0
  - **Score: 60.4 / 75**
- **Comparison:** Top-3 matches the `gpt-4o-mini` run, while top-1 improves from 0.567 to 0.783 and the score rises from 49.6 to 60.4 / 75. It had changed 67% of the retrieved rankings in the top-3
- **Conclusion:** `gpt-4.1-mini` is the strongest measured reranker so far on the development set. Results are limited to these 60 notes.

- **Follow-up results (reranking candidate pool increased to 50, 60 dev notes, official `score.py`):**
  - Slice A (n=60)
  - Top-1 accuracy: 0.783
  - Top-3 accuracy: 0.867
  - Points: 60.8 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0
  - **Score: 60.8 / 75**
- **Comparison with the previous `gpt-4.1-mini` run:** Top-1 stayed at 0.783; top-3 increased from 0.850 to 0.867, and the score rose from 60.4 to 60.8 / 75.

- **Observation :**
  - even though there is a marginal improvement, the cost increased is ~27% because of 50 retrieved candidates being used for re-ranking instead of 30.

- **Improvements needed :**
 - Had to increase the top-3 accuracy to 1.00 so that they can be reranked to top-1 for better score
 - To keep the most relevant code in top-1 

- **Alternatives Considered :**
  - using the tree-hierarchy mms-data-2026 of diseases and codes to improve ranking
  - prompt engineering by using fewshot, CoT
  - using a larger model than gpt-4.1-mini

## Decision 16: Few-shot (3 demostrations) + CoT for better task understanding

- **Observation:** There is no increase in the LLM's performance from these prompt engineering techniques (few-shot examples and CoT reasoning). The score and top-1/top-3 accuracy did not improve over the previous rerank prompt.

- **Alternatives Considered :**
  - using the tree-hierarchy mms-data-2026 of diseases and codes to improve ranking
  - using a larger model than gpt-4.1-mini


## Decision 17: Using the multi-lingual embedding model

- **Model:** `andorei/BERGAMOT-multilingual-GAT`
- **Results (60 notes):**
  - Slice A (n=60)
  - Top-1 accuracy: 0.617
  - Top-3 accuracy: 0.683
  - Points: 47.9 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0
  - **Score: 47.9 / 75**

- **Model:** `FremyCompany/BioLORD-2023-M`
- **Results (60 notes):**
  - Slice A (n=60)
  - Top-1 accuracy: 0.750
  - Top-3 accuracy: 0.783
  - Points: 57.1 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0
  - **Score: 57.1 / 75**

- **Follow-up results (50 candidates, 60 notes):**
  - Slice A (n=60)
  - Top-1 accuracy: 0.717
  - Top-3 accuracy: 0.783
  - Points: 55.4 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0
  - **Score: 55.4 / 75**

- **Observation:** Increasing the candidate pool from 30 to 50 did not improve the result: top-3 accuracy stayed at 0.783, while top-1 fell from 0.750 to 0.717 and the score decreased from 57.1 to 55.4 / 75.

- **Alternative Considered :**
  - use the tree hierarchy knowledge

## Decision 18 : Use the tree hierarchy knowledge for better re-ranking

- **Change (Stage 5, `src/coder.py`):**
  - Recall fallback: the LLM names the ICD-11 category of the main diagnosis; leaves of that category that the retriever missed (up to 20) are added and reranked.
  - Specific-child rule: a non-leaf pick becomes `<parent>.Z` unless the note supports a named child.
  - Prompt: the LLM first checks which candidates fit the main diagnosis, then ranks them, so the first pick is the most relevant. The output JSON adds `category`.
  - Dropped on the gate: residual-sibling rule (hurt one held-out fold, helped neither) and slot-2 strategy (hurt the even fold).

- **Results (60 notes), current code, OpenAI gpt-4.1-mini, `temperature=0`:**
  - Slice A (n=60)
  - Top-1 accuracy: 0.767
  - Top-3 accuracy: 0.833
  - Points: 59.2 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0
  - **Score: 59.2 / 75**

- **Run-to-run variance (same code):**
  - Unseeded runs: 57.9, 60.0, 60.0, 60.4 (`predictions/llm_run_stage5_post.jsonl` scored 57.9)
  - With `seed=0` added to the OpenAI request: 61.7, 61.7
  - Eleven of 60 notes changed between two unseeded runs, so the spread (about 4 points) is larger than most of the gains we have measured.
  - Baseline before Stage 5: 57.9 / 75.

- **Observation:**
  - The ablation favoured keeping the recall fallback and the specific-child rule. The per-fold deltas are small (n=30 per fold), so none of the gains is clearly significant.
  - Top-3 (0.833) is close to the top-3 ceiling of the retriever-plus-LLM pipeline, so the remaining gap is mostly in top-1 (0.767). Top-1 is worth 2× in the score.
  - The LLM's picks are not stable at `temperature=0`. Seeding helps but does not guarantee identical output. Keeping the residual-sibling and slot-2 rules (dropped on the gate).

- **Next steps to try to exceed 65 / 75:**
  - Measure the mean over several seeded runs before judging any change; a single run is not enough.
  - Self-consistency: run the rerank 3 to 5 times and take the majority top-1. This targets the instability directly and is cheap with gpt-4.1-mini.
  - Error analysis on the top-1 misses: group them by type (wrong category, wrong specificity, residual vs named child) and fix the most common type.
  - Try a stronger model (for example gpt-4.1 or Claude Sonnet) on the same pipeline, and check cost against the $1 budget first.
  - Improve retrieval recall for the category the LLM names, so the right leaf is in the candidate list before reranking.

- **Alternatives Considered :**
  - Larger model without changing the prompt (Decision 16 showed prompt changes alone did not help).

## Decision 19: Trying gpt-5.4-mini as the Stage 5 reranker

- **Change:** Same Stage 5 pipeline as Decision 18 (hybrid retriever, recall fallback, specific-child rule, `category` in output), with the OpenAI reranker switched to `gpt-5.4-mini`. Predictions: `predictions/llm_run_stage5_post_5.4.jsonl`.

- **Results (60 notes), official `score.py`:**
  - Slice A (n=60)
  - Top-1 accuracy: 0.733
  - Top-3 accuracy: 0.883
  - Points: 58.7 / 75
  - Missing predictions: 0
  - Invalid codes: 0
  - Duplicate IDs: 0
  - **Score: 58.7 / 75**

- **Comparison with gpt-4.1-mini (Decision 18, 59.2 / 75):**
  - Top-1 fell from 0.767 to 0.733, so the score is about the same, but top-3 rose from 0.833 to 0.883.
  - The score gap (0.5 points) is smaller than the run-to-run spread measured in Decision 18 (about 4 points), so this is not a measured improvement.

- **Observation:**
  - A stronger or newer model did not fix top-1. The model finds the right code in its top 3 more often, but it does not put the right code first.
  - Top-1 is worth 2× in the score, so the gap that matters is ranking within the top 3, not recall.
  - Changing the model alone does not change the results. The evaluation method (single unseeded run per model, 60 notes) cannot separate models that differ by about 1 to 4 points, so the method has to change before the next model comparison.

- **Methodology change needed:**
  - Measure each configuration as a mean over several seeded runs (at least 3), and report the spread, not a single score.
  - Evaluate the top-1 misses separately (wrong category, wrong specificity, residual vs named child) before trying another model.
  - Use a larger dev set or the 30-note folds from Decision 18 with paired comparisons, so small differences are tested rather than read from one run.
  - Keep the score as the headline, but judge a change only if it beats the run-to-run spread of the baseline.

- **Alternatives Considered :**
  - Swapping in a larger model without changing the method (this is what this decision did; it did not separate the models).

## Decision 20: Self-consistency and residual-vs-named prompt rule for the Stage 5 rerank

- **Change (`src/coder.py`):**
  - Self-consistency: the code supports a majority vote over `SELF_CONSISTENCY_SAMPLES` rerank passes (first pass greedy, `temperature=0`, `seed=0`; the rest sample at `SAMPLE_TEMPERATURE = 0.7`). The default is set to `1` (greedy only), because the 3-sample vote costs about 3x the LLM spend and did not beat greedy on its own (see results).
  - Prompt step 5: a named child code needs its feature stated in the note. When the note gives only the general diagnosis, the `.Z` code is correct even if a named child also fits.

- **Error analysis (Decision 18 run, 14 top-1 misses, `misc/llm_run_stage5_post_per_note.jsonl`):**
  - Residual vs named child: 8 (gold is `.Z`, the pick is a named leaf or a different `.Z`). Gold was in the 30-candidate pool in all 8, so this is a ranking problem.
  - Wrong category: 6. The gold category was not in the retriever pool for any of them, so only the recall fallback can help, and it helps only when the LLM names the right category.
  - Wrong leaf within the right category: 1 (`CA40.Z` vs `CA40.0Z`).
  - Gold is a `.Z` code in 14 notes; top-1 is right in only 5 of them.

- **Results (60 notes, single runs, `score.py`):**

| Configuration | Top-1 | Top-3 | Score |
|---|---|---|---|
| Baseline (current code, seed 0) | 0.783 | 0.883 | 61.2 / 75 |
| Self-consistency (3) only | 0.783 | 0.867 | 60.8 / 75 |
| Residual prompt only | 0.817 | 0.850 | 62.1 / 75 |
| Residual prompt + self-consistency (3) | 0.817 | 0.867 | 62.5 / 75 |
| Shipped config (`SELF_CONSISTENCY_SAMPLES = 1`, residual prompt), rerun 1 | 0.800 | 0.850 | 61.2 / 75 |
| Shipped config (`SELF_CONSISTENCY_SAMPLES = 1`, residual prompt), rerun 2 | 0.833 | 0.867 | 63.3 / 75 |

Rerun 2 cost $0.0847 in LLM spend, the same as rerun 1.

- **Observation:**
  - The residual prompt fixed 3 of the 8 residual misses (dev-014, dev-022, dev-023) and broke 2 top-3 hits (dev-007 now picks `CA40.0Z`, dev-057 picks `5A00.Z`).
  - Self-consistency alone did not help. Most votes agree with the greedy pass, and it costs about 3× the LLM spend ($0.26 vs $0.08 per 60-note run).
  - The total gain (+1.3 points) is about one note in top-1 and is inside the run-to-run spread from Decision 18 (about 4 points). It is not a measured improvement yet.

- **Next steps:**
  - Repeat the baseline and the combined config over at least 3 runs each before keeping the change.
  - Recall for the named category (wrong-category misses): the gold category is outside the pool, so the next step is to retrieve more leaves for the named category, or to log the named category per note to see how often it is the gold one.
  - Budget: with 3 samples, the $0.85 per-run guard in `src/coder.py` is reached after about 200 notes, after which the rest keep the retriever ranking. The default is 1 for that reason.
  - Chosen default: `SELF_CONSISTENCY_SAMPLES = 1` (user decision). Two reruns of this exact configuration gave 61.2 and 63.3, not the 62.1 from the first residual-prompt run. That is a spread of about 2 points on identical settings, so greedy decoding is not fully deterministic across runs, and every difference in this table is within noise. The residual prompt change is not shown to help.

- **Alternatives Considered :**
  - Self-consistency with temperature 0 only: rejected, because identical greedy calls give identical votes.

## Decision 21: Diagnosing the top-3 misses before the next change

- **Date:** 2026-11-12 15:55 IST.
- **Commit:** added in the commit that includes this entry (parent `fe3ae34`).

- **Context:** The run in `misc/llm_run_sc_per_note.jsonl` (shipped config, `SELF_CONSISTENCY_SAMPLES = 1`, residual prompt) scores top-1 0.833 (50/60), top-3 0.867 (52/60), 63.3 / 75. Top-3 needs 54/60 to reach 0.90.

- **Diagnosis (8 top-3 misses):**

| Note | Gold | Predicted | Type |
|---|---|---|---|
| dev-002 | CA00, CA07.0 (viral URTI) | 1E1Z, 9B72.0Z, 1D86 | Wrong category, vocabulary |
| dev-008 | 1G05 (jiggers) | 1G0Z, 1G04.Z, 1G04.Y | Right block, wrong leaf, vocabulary |
| dev-028 | DA22.Z | DD90.2, 9B70, LD2H.4 | Wrong category |
| dev-033 | FA25.2Z | FA25.0, FA25.20, FA25 | Right family, wrong specificity |
| dev-051 | 5B52 | MG29.3, CB01, MG29 | Wrong category, symptom codes |
| dev-056 | JA63.2 | QA43.Z, 5B81.01, 5B80.01 | Wrong category |
| dev-057 | 5A00.2Z | 5A00.22, 5A00.01, 5A00 | Right family, wrong specificity |
| dev-060 | 5B81.01 | 5A0Z, 5A0Y, 5A02.3 | Wrong category |

- **Summary:** 5 of 8 are wrong category; 3 are right family with wrong specificity.

- **Pool check (done, rebuilt with `retrieve_candidates` and the default embedding model):**
  - The wrong picks are all in the 30-candidate pool, so the LLM chooses from the pool correctly and the pool is not the cause of those picks.
  - The gold code is **not** in the pool for 6 of the 8 misses (dev-002, dev-008, dev-028, dev-051, dev-056, dev-060). A rerank cannot put a code in the top 3 that the pool does not contain. These are retrieval (vocabulary) misses.
  - The gold is in the pool for 2 of the 8: dev-033 (`FA25.2Z` at rank 6) and dev-057 (`5A00.2Z` at rank 14). These are specificity misses.
  - Correction: my first table above has the wrong-category label for dev-051, dev-056 and dev-060 as a reranking problem. The gold is absent from the pool for those notes, so it is a retrieval problem.

- **Cause found in the code:** `specific_code` in `src/coder.py` turned a non-leaf pick into its best-ranked named child in pool order, before trying the unspecified `.Z` child. Decision 18 documents the rule as "a non-leaf pick becomes `<parent>.Z`". The code did not follow that rule.

- **Change:** `specific_code` now returns the `.Z` child for a non-leaf pick, and falls back to the best-ranked named child only when no `.Z` child exists. Nothing else changed.

- **Results (60 dev notes, `score.py`, same model `gpt-4.1-mini`, three full runs):**

| Run | Top-1 | Top-3 | Score |
|---|---|---|---|
| Fixed rule, run 1 | 0.883 | 0.883 | 66.2 / 75 |
| Fixed rule, run 2 | 0.850 | 0.867 | 64.2 / 75 |
| Fixed rule, run 3 | 0.900 | 0.900 | 67.5 / 75 |
| Previous shipped config, reruns 1 and 2 (Decision 20) | 0.800 / 0.833 | 0.850 / 0.867 | 61.2 / 63.3 |

  - Mean of the three fixed runs: 66.0 / 75 (top-1 0.878, top-3 0.883), against 62.3 for the previous config.
  - Cost: about $0.08 per 60-note run.
  - Caveat: three runs, a spread of about 3 points, and the dev set is also the set used for tuning. This is a measured gain on dev, not proof of a hidden-set gain.
  - Top-3 misses per run: run 1 = dev-002, 008, 028, 033, 051, 056, 060; run 2 = those 7 plus dev-042; run 3 = dev-002, 008, 028, 051, 056, 060. The six notes that miss in all three runs are the vocabulary misses above. dev-033 (specificity) was fixed in run 3 only.

- **Options considered:**
  1. Check candidate-pool recall for these 8 notes first. Decides whether the wrong-category misses are retrieval or ranking problems.
  2. LLM term extraction before retrieval (plan Stage 1B). Targets dev-002, dev-008, dev-056 (vocabulary gaps).
  3. Specificity handling for the family misses (dev-033, dev-057). Any rule must be general, not written for one note.
  4. More self-consistency samples to fill top-3. Rejected as default: 3 samples cost about $0.26 per 60 notes, about $1.30 per 300 notes, which exceeds the $1 budget. Self-consistency ranks by vote, so it mainly helps top-1.

- **Decision:** Step 1 (pool check) first, as agreed. Pool check result: the gold is missing from the pool for 6 of 8 misses, so reranking cannot fix them. The one generic code fault found (non-leaf pick rule, see Pool check above) is fixed and kept, because it follows the documented rule and improved the three-run mean. Option 2 (term extraction before retrieval, plan Stage 1B) is the next step for the 6 vocabulary misses, and is not done yet.

- **Observation:** The gap to 90% on dev is 2 notes. Dev is also the set used for tuning, so a 90% dev top-3 is not evidence the hidden score will reach it. Hidden-set generalisation (Stage 6) matters more than these 2 notes.

- **Revisit:** Option 2 (LLM term extraction before retrieval) for the 6 vocabulary misses, with a check that it does not add new top-3 misses. Re-check the budget if the sample count changes. Repeat the fixed config over more runs before calling it the shipped default.

- **Alternatives Considered :**
- Check pool recall for these 8 notes. Log whether the gold is in the 30-candidate pool. If it isn't, the fix is retrieval, not reranking.
- Term extraction before retrieval. Have the LLM extract the diagnosis as English clinical terms and retrieve with those too. This targets dev-002, dev-008 and dev-056.
- Specificity handling for the family misses. Look at how .Z and named children are ranked in the prompt for dev-033 and dev-057. Keep any rule general, not written for one note. ( this is for 3 family but specificity misses)

- **Results :**

  - **Slice A (60 dev notes, `score.py`):** top-1 0.867 (52/60), top-3 0.867 (52/60), **Score: 65.0 / 75**. Validation: 60 notes, 0 missing predictions, 0 invalid codes, 0 duplicate ids.
  - Comparison: 65.0 is within the spread of the three fixed-rule runs above (64.2 to 67.5, mean 66.0). Top-3 of 0.867 is below the 0.900 of the best fixed run.

## Decision 22: Term extraction and abbreviation expansion for the vocabulary misses

- **Context:** Decision 21 found that the gold code is missing from the 30-candidate pool for 6 of the 8 top-3 misses (dev-002, dev-008, dev-028, dev-051, dev-056, dev-060). Reranking cannot fix these, so the fix must be in retrieval. The causes are shorthand in the note (for example GORD, SAM, URTI) and lay or paraphrased terms that do not match catalogue titles.

- **Change (generic, no note-specific text in prompts):**
  1. **Term extraction (plan Stage 1B):** one short LLM call per note (`extract_terms`, prompt `TERMS_SYSTEM`) returns up to 4 English clinical diagnosis terms, spelled out, with lay words translated. The terms lead the retrieval query. If the call fails, the note gets no terms. Controlled by `TERM_EXTRACTION`.
  2. **Abbreviation expansion:** `QUERY_ABBREVIATIONS` (about 25 standard clinical abbreviations) is applied to the retrieval query only, not to the LLM prompt.

- **Pool check (abbreviations only):** gold in the pool for 54 of 61 dev gold codes, up from 53 of 61. dev-028 (GORD) entered the pool at rank 23. I did not re-run the pool check after adding term extraction.

- **Results (60 dev notes, `score.py`, `gpt-4.1-mini`):**

| Run | Top-1 | Top-3 | Score |
|---|---|---|---|
| Term extraction + abbreviations, run 1 | 0.917 | 0.967 | 70.0 / 75 |
| Term extraction + abbreviations, run 2 | 0.933 | 0.967 | 70.8 / 75 |
| Term extraction + abbreviations, run 3 | 0.933 | 0.967 | 70.8 / 75 |
| Fixed rule only, mean of 3 runs (Decision 21) | 0.878 | 0.883 | 66.0 / 75 |

  - Mean of the 3 term-extraction runs: 70.5 / 75 (top-1 0.928, top-3 0.967), up from 66.0.
  - Top-3 misses are the same in all 3 runs: dev-002 (gold CA00, CA07.0; predicted 1E1Z, 1D86, 1D4Z) and dev-051 (gold 5B52; predicted MG29.01/MG29.3/CB01, or MG29.3/CB01/EH40.3 in run 2). These are the 2 notes that remain.
  - Cost: about $0.093 per 60-note run. Linear scaling gives about $0.47 for 300 notes, within the $1 budget. Only the term call is added to the pipeline.

- **Caveats:**
  - Dev is also the tuning set. The gain is measured on dev only.
  - The abbreviation list was written after reading the dev misses (GORD, SAM, URTI came from them). It is a standard clinical list, but it is not independent of dev.
  - Slice B (75 hidden notes) is not available locally, so generalisation is not measured. The changes are generic, but this is unverified.

