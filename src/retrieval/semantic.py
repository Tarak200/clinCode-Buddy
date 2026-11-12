"""Semantic retriever for ICD-11 codes using embeddings and similarity matching."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Optional
import pickle
import warnings

import numpy as np


@dataclass(frozen=True)
class Candidate:
    code: str
    title: str
    score: float


class SemanticIndex:
    """Rank catalogue titles using semantic similarity with precomputed embeddings."""

    def __init__(
        self,
        records: list[dict[str, object]],
        embedder=None,
        embeddings_cache_path: Optional[Path] = None,
        show_progress_bar: bool = False,
    ) -> None:
        if not records:
            raise ValueError("The MMS catalogue is empty")

        self._documents: list[tuple[str, str]] = []
        self._embeddings: Optional[np.ndarray] = None
        seen_codes: set[str] = set()
        self._embedder = embedder

        for record in records:
            code = record.get("code")
            title = record.get("title")
            if not isinstance(code, str) or not code.strip():
                raise ValueError("Every catalogue entry must have a nonempty code")
            if not isinstance(title, str) or not title.strip():
                raise ValueError(f"Catalogue entry {code!r} has no title")
            if code in seen_codes:
                raise ValueError(f"Duplicate catalogue code: {code}")
            seen_codes.add(code)

            self._documents.append((code, title))

        model_name = getattr(self._embedder, "model_name_or_path", None)
        if not isinstance(model_name, str):
            model_name = (
                f"{type(self._embedder).__module__}."
                f"{type(self._embedder).__qualname__}"
            )
        cache_fingerprint = hashlib.sha256(
            json.dumps(
                {"documents": self._documents, "model": model_name},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        if embeddings_cache_path and embeddings_cache_path.exists():
            try:
                with open(embeddings_cache_path, 'rb') as f:
                    cached = pickle.load(f)
            except (
                OSError,
                EOFError,
                pickle.UnpicklingError,
                AttributeError,
                ImportError,
                ValueError,
                TypeError,
            ) as error:
                warnings.warn(
                    f"Could not load embeddings cache {embeddings_cache_path}: {error}",
                    RuntimeWarning,
                )
            else:
                if (
                    isinstance(cached, dict)
                    and cached.get("fingerprint") == cache_fingerprint
                ):
                    try:
                        cached_embeddings = np.asarray(cached.get("embeddings"))
                    except (TypeError, ValueError):
                        cached_embeddings = np.asarray([])
                    if (
                        cached_embeddings.ndim == 2
                        and cached_embeddings.shape[0] == len(self._documents)
                        and np.issubdtype(cached_embeddings.dtype, np.number)
                        and np.isfinite(cached_embeddings).all()
                    ):
                        self._embeddings = cached_embeddings
                        if show_progress_bar:
                            print(
                                f"Loaded cached embeddings for {len(self._documents)} catalogue titles",
                                flush=True,
                            )
                    else:
                        warnings.warn(
                            f"Embeddings cache {embeddings_cache_path} has invalid data; recomputing",
                            RuntimeWarning,
                        )
                else:
                    warnings.warn(
                        f"Embeddings cache {embeddings_cache_path} does not match this catalogue or model; recomputing",
                        RuntimeWarning,
                    )

        if self._embeddings is None and self._embedder:
            titles = [title for _, title in self._documents]
            if hasattr(self._embedder, "get_sentence_embedding_dimension"):
                if show_progress_bar:
                    print(
                        f"Embedding {len(titles)} catalogue titles (batch size 64)...",
                        flush=True,
                    )
                embeddings_list = self._embedder.encode(
                    titles,
                    batch_size=64,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress_bar,
                )
            else:
                if show_progress_bar:
                    print(f"Embedding {len(titles)} catalogue titles...", flush=True)
                embeddings_list = [self._embedder.encode(title) for title in titles]
            embeddings = np.asarray(embeddings_list)
            if (
                embeddings.ndim != 2
                or embeddings.shape[0] != len(self._documents)
                or not np.issubdtype(embeddings.dtype, np.number)
                or not np.isfinite(embeddings).all()
            ):
                raise ValueError("Embedder returned invalid catalogue embeddings")
            self._embeddings = embeddings

            if embeddings_cache_path:
                embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(embeddings_cache_path, 'wb') as f:
                    pickle.dump(
                        {
                            "fingerprint": cache_fingerprint,
                            "embeddings": self._embeddings,
                        },
                        f,
                    )

    @classmethod
    def from_file(
        cls,
        path: Path,
        embedder=None,
        embeddings_cache_path: Optional[Path] = None,
        show_progress_bar: bool = False,
    ) -> "SemanticIndex":
        with path.open(encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records
        ):
            raise ValueError("The MMS catalogue must be a JSON array of objects")
        return cls(
            records,
            embedder=embedder,
            embeddings_cache_path=embeddings_cache_path,
            show_progress_bar=show_progress_bar,
        )

    def search(self, text: str, limit: int = 50) -> list[Candidate]:
        if limit < 1:
            raise ValueError("The candidate limit must be at least 1")
        
        if not self._embedder or self._embeddings is None or not self._embeddings.size:
            raise ValueError("Embedder not initialized or embeddings not computed")

        query_embedding = np.asarray(self._embedder.encode(text))
        if (
            query_embedding.ndim != 1
            or query_embedding.shape[0] != self._embeddings.shape[1]
            or not np.issubdtype(query_embedding.dtype, np.number)
            or not np.isfinite(query_embedding).all()
        ):
            raise ValueError("Embedder returned an invalid query embedding")
        
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8)
        scores = doc_norms @ query_norm

        # Ties are broken by code so the ranking is deterministic.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), self._documents[index][0]),
        )
        
        candidates = []
        for index in ranked_indices[:limit]:
            code, title = self._documents[int(index)]
            score = float(scores[index])
            candidates.append(Candidate(code=code, title=title, score=score))
        
        return candidates
