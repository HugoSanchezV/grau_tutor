"""Índice léxico BM25 + fusión RRF para retrieval híbrido."""
from __future__ import annotations
import os
import pickle
import re
from typing import Sequence
from rank_bm25 import BM25Okapi
from contracts.partida import PartidaGrau
from core.logging import get_logger

logger = get_logger(__name__)

_SPANISH_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "a", "al", "en", "con", "por", "para", "sin", "sobre",
    "que", "como", "y", "o", "u", "e", "pero", "si", "no", "se",
    "le", "lo", "les", "me", "te", "nos", "su", "sus", "mi", "mis",
    "es", "son", "era", "fue", "ser", "ha", "han", "había",
    "este", "esta", "esto", "estos", "estas",
    "ese", "esa", "eso", "esos", "esas",
    "aquel", "aquella", "aquello",
    "más", "muy", "también", "ya", "solo", "sólo", "aún", "tanto",
    "cuando", "donde", "cual", "cuales", "quien", "quienes",
    "porque", "entre", "desde", "hasta", "hacia", "durante",
    "ante", "tras", "según", "contra",
}

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [
        tok
        for tok in (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
        if len(tok) > 1 and tok not in _SPANISH_STOPWORDS
    ]


def _doc_text_for_bm25(chunk: PartidaGrau) -> str:
    meta = chunk.metadata
    parts = [meta.tema, meta.eco, meta.white, meta.black, chunk.comentarios]
    return " ".join(p for p in parts if p)


def build_bm25_index(chunks: Sequence[PartidaGrau], path: str) -> None:
    if not chunks:
        logger.warning("No hay chunks para BM25; índice no creado.")
        return
    ids = [c.partida_id for c in chunks]
    tokenized_corpus = [_tokenize(_doc_text_for_bm25(c)) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"ids": ids, "bm25": bm25}, f)
    logger.info(f"Índice BM25 guardado en {path} ({len(ids)} docs)")


def load_bm25_index(path: str) -> dict | None:
    if not os.path.exists(path):
        logger.warning(f"Índice BM25 no encontrado en {path}; retrieval será solo denso.")
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Índice BM25 cargado desde {path} ({len(data['ids'])} docs)")
    return data


def bm25_search(index_data: dict, query: str, n: int = 20) -> list[tuple[str, float]]:
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = index_data["bm25"].get_scores(tokens)
    ids = index_data["ids"]
    ranked = [(id_, float(s)) for id_, s in zip(ids, scores) if s > 0]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:n]


def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
    top_n: int = 5,
) -> list[str]:
    """Reciprocal Rank Fusion. k=60 es el valor canónico (Cormack et al., 2009)."""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in fused[:top_n]]
