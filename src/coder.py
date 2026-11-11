"""Generate ICD-11 predictions from JSONL notes.

The RRF retriever fetches candidates. An LLM then extracts the diagnoses from the
note (4a) and picks up to 3 codes from the candidates (4b). The LLM is OpenAI if
OPENAI_API_KEY is set, else Anthropic if ANTHROPIC_API_KEY is set, else the local
Qwen model. Any LLM problem falls back to the retriever ranking for that note.

Flags:
  --hybrid  retriever (RRF) only, no LLM is loaded or called
  --api     use the OpenAI/Anthropic API; falls back to local Qwen if no key or library
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from src.retrieval.hybrid import HybridIndex
from tqdm import tqdm

BUDGET_USD = 0.85  # hard stop, the real limit is $1 for the whole run
CALL_TIMEOUT_S = 30
WORKERS = 8
EMBEDDING_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
CANDIDATES = 30
MAX_NOTE_CHARS = 8000
# (input, output) USD per 1M tokens
PRICES = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-haiku-4-5": (1.00, 5.00),
}
DEFAULT_MODELS = {"openai": "gpt-4.1-mini", "anthropic": "claude-haiku-4-5"}
# Local model run through transformers; weights are downloaded once into LOCAL_MODEL_CACHE
LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_MODEL_CACHE = Path(__file__).resolve().parent.parent / "data" / ".model_cache"
LOCAL_MAX_NEW_TOKENS = 100  # the JSON answers are short; longer output only adds latency

EXTRACT_SYSTEM = """You read a clinical consultation note (any language, possibly a transcript).
Return JSON only:
{"current_diagnoses": [English diagnosis terms the patient currently has],
 "ruled_out": [English terms that are negated, ruled out, or only in the differential],
 "language": "...",
 "is_specific_enough": true/false}
Ignore past history that is resolved. Include diagnoses mentioned by a caregiver or in the Q&A.
If no diagnosis is stated, list the main symptoms as current_diagnoses.
is_specific_enough is false when the note lacks the detail (type, site, cause) to pick a specific code."""

RERANK_SYSTEM = """You assign ICD-11 MMS codes to a clinical note.
Choose up to 3 codes ONLY from the candidate list, best first. Return JSON only: {"codes": ["...", "..."]}
Rules:
- Use the most specific code the note supports; never a parent when a child fits.
- When the detail needed for a specific code is absent, prefer the unspecified code (code ending in .Z).
- Never code a condition that is ruled out or only in the differential.
- Code symptoms only if the note gives no diagnosis.
- The first code must be the main current diagnosis. The ranking of codes in the output json should reflect the relevance of each code to the note."""


class BudgetExceeded(Exception):
    pass


class LLM:
    """Small JSON-in/JSON-out wrapper over the OpenAI or Anthropic SDK with a cost cap."""

    def __init__(self, provider: str, client, model: str) -> None:
        self.provider = provider
        self.client = client  # SDK client, or (tokenizer, model) for "local"
        self.model = model
        self.spent = 0.0
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()  # one local generation at a time

    def _price(self, tokens_in: int, tokens_out: int) -> float:
        if self.provider == "local":
            return 0.0
        price_in, price_out = PRICES.get(self.model, (1.0, 5.0))  # unknown model: assume dear
        return (tokens_in * price_in + tokens_out * price_out) / 1_000_000

    def _call_local(self, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        import torch

        tokenizer, model = self.client
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with self._generate_lock, torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, LOCAL_MAX_NEW_TOKENS),
                do_sample=False,
                stop_strings=["}"],  # the JSON ends at the first closing brace
                tokenizer=tokenizer,
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, prompt_len, len(new_tokens)

    def _call(self, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        if self.provider == "local":
            return self._call_local(system, user, max_tokens)
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
            )
            usage = response.usage
            return response.choices[0].message.content or "", usage.prompt_tokens, usage.completion_tokens
        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0,
        )
        usage = response.usage
        return response.content[0].text, usage.input_tokens, usage.output_tokens

    def ask_json(self, system: str, user: str, max_tokens: int = 300) -> dict:
        """Return the parsed JSON answer. Tries twice; raises on failure or budget."""
        worst_case = self._price(len(system + user) // 3, max_tokens)  # ~3 chars/token
        last_error: Exception | None = None
        for _ in range(2):
            with self._lock:
                if self.spent + worst_case > BUDGET_USD:
                    raise BudgetExceeded(f"budget of ${BUDGET_USD} reached")
            try:
                text, tokens_in, tokens_out = self._call(system, user, max_tokens)
            except Exception as error:
                last_error = error
                continue
            with self._lock:
                self.spent += self._price(tokens_in, tokens_out)
            start, end = text.find("{"), text.rfind("}")
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError as error:
                last_error = error
        raise RuntimeError(f"LLM call failed: {last_error}")


def load_env(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file without overriding real environment variables."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip().strip("'\"")
            if value:
                os.environ.setdefault(key.strip(), value)


def make_llm() -> LLM | None:
    """Use OpenAI if its key is set, else Anthropic, else the local Qwen model.

    A missing API library or key moves on to the next option instead of giving up.
    """
    model_override = os.environ.get("LLM_MODEL")
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI(timeout=CALL_TIMEOUT_S, max_retries=0)  # honours HTTPS_PROXY
            model = model_override if model_override and not model_override.startswith("claude") else DEFAULT_MODELS["openai"]
            return LLM("openai", client, model)
        except ImportError as error:
            print(f"openai library missing ({error}); trying next LLM", file=sys.stderr)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from anthropic import Anthropic

            client = Anthropic(timeout=CALL_TIMEOUT_S, max_retries=0)
            model = model_override if model_override and model_override.startswith("claude") else DEFAULT_MODELS["anthropic"]
            return LLM("anthropic", client, model)
        except ImportError as error:
            print(f"anthropic library missing ({error}); trying next LLM", file=sys.stderr)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        torch.set_num_threads(os.cpu_count() or 1)  # use all CPUs for generation

        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, cache_dir=LOCAL_MODEL_CACHE)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, cache_dir=LOCAL_MODEL_CACHE, torch_dtype="auto")
        model.eval()
        return LLM("local", (tokenizer, model), LOCAL_MODEL)
    except Exception as error:  # missing libraries, download or load failures: use the retriever alone
        print(f"local LLM unavailable ({error}); using retriever only", file=sys.stderr)
    return None


def llm_codes(llm: LLM, index, lock: threading.Lock, text: str, valid: dict[str, str]) -> list[str]:
    """4a extract -> retrieve -> 4b rerank. Returns up to 3 valid codes."""
    text = text[:MAX_NOTE_CHARS]

    # 4a: extract diagnoses (failure is fine, we just retrieve with the note alone)
    try:
        info = llm.ask_json(EXTRACT_SYSTEM, text, max_tokens=300)
    except BudgetExceeded:
        raise
    except Exception:
        info = {}
    diagnoses = [str(term) for term in info.get("current_diagnoses") or []]
    ruled_out = [str(term) for term in info.get("ruled_out") or []]

    # Retrieve with the note plus the English terms, and with the terms alone
    with lock:  # the embedder is shared between workers
        found = index.search(text + "\n" + "; ".join(diagnoses), limit=CANDIDATES)
        if diagnoses:
            found += index.search("; ".join(diagnoses), limit=CANDIDATES // 2)
    candidates: dict[str, str] = {}
    for item in found:
        if item.code in valid and item.code not in candidates and len(candidates) < CANDIDATES:
            candidates[item.code] = item.title

    # 4b: rerank, picking only from the candidate list
    lines = []
    for code, title in candidates.items():
        parent = valid.get(code.split(".")[0]) if "." in code else None
        lines.append(f"{code} | {title}" + (f" | parent: {parent}" if parent else ""))
    prompt = (
        f"NOTE:\n{text}\n\n"
        f"Current diagnoses (extracted): {diagnoses}\n"
        f"Ruled out: {ruled_out}\n"
        f"Enough detail for a specific code: {info.get('is_specific_enough', 'unknown')}\n\n"
        "CANDIDATES (code | title | parent):\n" + "\n".join(lines)
    )
    answer = llm.ask_json(RERANK_SYSTEM, prompt, max_tokens=100)

    chosen: list[str] = []
    for code in answer.get("codes") or []:
        if code in candidates and code not in chosen:
            chosen.append(code)
    # Pad with the retriever order so top-3 still has three guesses
    for code in candidates:
        if len(chosen) >= 3:
            break
        if code not in chosen:
            chosen.append(code)
    return chosen[:3]


def run(
    input_path: Path,
    output_path: Path,
    catalogue_path: Path,
    hybrid: bool = False,
) -> None:
    paths = [input_path.resolve(), output_path.resolve(), catalogue_path.resolve()]
    if len(set(paths)) != len(paths):
        raise ValueError("Input, output, and catalogue paths must differ")

    # No API key and no local model available -> llm is None and the RRF retriever alone is used
    load_env(Path(__file__).resolve().parent.parent / ".env")
    llm = None if hybrid else make_llm()
    if hybrid:
        print("Hybrid mode: retriever only, no LLM", file=sys.stderr)
    elif llm:
        print(f"Using {llm.provider} model {llm.model} on top of the retriever", file=sys.stderr)
    else:
        print("No LLM key found; using the retriever only", file=sys.stderr)
    with catalogue_path.open(encoding="utf-8") as file:
        valid = {record["code"]: record["title"] for record in json.load(file)}

    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers is required for retrieval. Install with: pip install sentence-transformers")

    print(f"Using Hybrid (RRF) Retriever with model: {EMBEDDING_MODEL}", file=sys.stderr)
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Cache the catalogue embeddings next to the catalogue
    cache_dir = catalogue_path.parent / ".embeddings_cache"
    cache_file = cache_dir / f"{catalogue_path.stem}_{EMBEDDING_MODEL.replace('/', '_')}.pkl"

    print("Building index...", file=sys.stderr)
    index = HybridIndex.from_file(
        catalogue_path,
        embedder=embedder,
        embeddings_cache_path=cache_file,
        show_progress_bar=True,
    )
    
    seen_ids: set[object] = set()
    notes: list[tuple[object, str]] = []

    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                note = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{input_path}:{line_number}: invalid JSON") from error
            if not isinstance(note, dict):
                raise ValueError(f"{input_path}:{line_number}: expected a JSON object")
            note_id = note.get("id")
            text = note.get("text")
            if "id" not in note or not isinstance(note_id, (str, int, float)):
                raise ValueError(f"{input_path}:{line_number}: id must be a string or number")
            if not isinstance(text, str):
                raise ValueError(f"{input_path}:{line_number}: text must be a string")
            if note_id in seen_ids:
                raise ValueError(f"{input_path}:{line_number}: duplicate input id {note_id!r}")
            seen_ids.add(note_id)
            notes.append((note_id, text))

    lock = threading.Lock()  # the shared index/embedder is not used concurrently
    budget_warned = False

    def predict(note: tuple[object, str]) -> dict:
        nonlocal budget_warned
        note_id, text = note
        codes: list[str] = []
        if llm:
            try:
                codes = llm_codes(llm, index, lock, text, valid)
            except BudgetExceeded as error:
                if not budget_warned:
                    budget_warned = True
                    print(f"{error}; using retriever only from here", file=sys.stderr)
            except Exception as error:
                print(f"note {note_id!r}: LLM failed ({error}); using retriever", file=sys.stderr)
        if not codes:
            with lock:
                codes = [item.code for item in index.search(text, limit=3)]
        return {"id": note_id, "codes": codes}

    print(f"Processing {len(notes)} notes from {input_path}", file=sys.stderr)
    # The local model serialises its own generation, so extra workers would only wait
    workers = WORKERS if llm and llm.provider != "local" else 1
    with output_path.open("w", encoding="utf-8") as output, ThreadPoolExecutor(
        workers
    ) as pool:
        # map yields in input order; each prediction is written as soon as it is ready
        for prediction in tqdm(pool.map(predict, notes), total=len(notes), desc="Coding notes", file=sys.stderr):
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            output.flush()

    if llm:
        print(f"LLM spend: ${llm.spent:.4f}", file=sys.stderr)
    print(f"Predictions written to {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Input JSONL notes",
        default=Path(__file__).resolve().parent.parent / "data" / "dev_notes.jsonl"
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output JSONL predictions",
        default=Path(__file__).resolve().parent.parent / "predictions" / "predictions.jsonl"
    )
    parser.add_argument(
        "--mms",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "mms-2026-01.json",
        help="MMS catalogue JSON file",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--hybrid",
        action="store_true",
        help="retriever (RRF) only, no LLM",
    )
    mode.add_argument(
        "--api",
        action="store_true",
        help="use the OpenAI/Anthropic API, falling back to local Qwen if unavailable (default behaviour)",
    )
    args = parser.parse_args()
    try:
        run(args.input, args.output, args.mms, hybrid=args.hybrid)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()



