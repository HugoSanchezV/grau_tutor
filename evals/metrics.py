"""Métricas de evaluación de retrieval. Funciones puras, sin estado."""
from __future__ import annotations
from statistics import mean


def hit_rate_at_k(relevance: list[bool], k: int) -> float:
    """1.0 si al menos un chunk relevante aparece en top-k; 0.0 si ninguno."""
    return 1.0 if any(relevance[:k]) else 0.0


def mrr_at_k(relevance: list[bool], k: int) -> float:
    """Reciprocal rank del primer chunk relevante dentro de top-k. 0 si ninguno."""
    for i, rel in enumerate(relevance[:k], start=1):
        if rel:
            return 1.0 / i
    return 0.0


def precision_at_k(relevance: list[bool], k: int) -> float:
    """Fracción de chunks relevantes dentro de top-k."""
    if k == 0:
        return 0.0
    top_k = relevance[:k]
    return sum(top_k) / k


def aggregate(per_query_scores: list[float]) -> float:
    """Promedio simple sobre queries."""
    if not per_query_scores:
        return 0.0
    return mean(per_query_scores)
