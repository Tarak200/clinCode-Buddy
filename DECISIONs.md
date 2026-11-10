## Decision 01: Fixing the basic plan to use for this problem

- **Date/time:** 10-Nov-2026, 14:17 IST
- **Commit:** None
- **Decision:** 
    The basic plan has been decided to start with the simple methodologies to complex based on the results and metrics
- **Reasoning:** As the task can be done in many ways, starting with simple methodology seems fair to understand where improvements are needed

## Decision 02: Implement the BM25 retriever baseline

- **Date/time:** 10-Nov-2026, 15:32 IST
- **Commit:** `TBD`
- **Decision:** Start with a dependency-free local BM25 index over the supplied MMS titles. I matched the words in description to the title in MMS data to return the three highest-ranked valid codes.
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



