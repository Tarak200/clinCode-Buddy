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
- **Commit:** `TBD`
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
  We can use the RRF + priors + assessment combination when we don't have any API available as a fallback







