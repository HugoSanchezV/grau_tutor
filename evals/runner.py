"""A/B eval del retriever: dense-only vs hybrid (dense + BM25 + RRF).

Ejecuta cada query del dataset contra ambos sistemas, evalúa cada chunk
recuperado con criterios verificables externos (keywords / metadata),
y reporta Hit@K, MRR@K y P@K.
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging import setup_logging, get_logger
from rag.store import get_chroma_client, get_or_create_collection
from rag.retrieval import GrauRetriever
from evals.metrics import hit_rate_at_k, mrr_at_k, precision_at_k, aggregate

setup_logging()
logger = get_logger(__name__)

K = 5
N_CANDIDATES = 20
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


def is_relevant(chunk: dict, criterio: dict) -> bool:
    tipo = criterio["tipo"]
    if tipo == "keyword_any":
        doc = chunk["doc"].lower()
        return any(term.lower() in doc for term in criterio["any_of"])
    if tipo == "keyword_all":
        doc = chunk["doc"].lower()
        return all(term.lower() in doc for term in criterio["all_of"])
    if tipo == "meta_any":
        meta = chunk["meta"]
        for cond in criterio["any_of"]:
            field_val = str(meta.get(cond["field"], ""))
            if cond["contains"].lower() in field_val.lower():
                return True
        return False
    raise ValueError(f"Tipo de criterio desconocido: {tipo}")


def evaluate_system(name: str, retriever: GrauRetriever, dataset: list[dict]) -> dict:
    per_query = []
    for q in dataset:
        chunks = retriever.retrieve_raw(q["query"], n_results=K)
        relevance = [is_relevant(c, q["criterio"]) for c in chunks]
        per_query.append({
            "id": q["id"],
            "query": q["query"],
            "categoria": q["categoria"],
            "relevance": relevance,
            "retrieved_ids": [c["id"] for c in chunks],
            "hit_rate": hit_rate_at_k(relevance, K),
            "mrr": mrr_at_k(relevance, K),
            "precision": precision_at_k(relevance, K),
        })
    return {
        "system": name,
        "per_query": per_query,
        "hit_rate": aggregate([r["hit_rate"] for r in per_query]),
        "mrr": aggregate([r["mrr"] for r in per_query]),
        "precision": aggregate([r["precision"] for r in per_query]),
    }


def print_summary(results: list[dict]) -> None:
    print()
    print("=" * 50)
    print(f"RESUMEN — k={K}, queries={len(results[0]['per_query'])}")
    print("=" * 50)
    print(f"{'Sistema':<15} {'Hit@K':>8} {'MRR@K':>8} {'P@K':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['system']:<15} {r['hit_rate']:>8.3f} {r['mrr']:>8.3f} {r['precision']:>8.3f}")
    print()


def print_per_query(results: list[dict]) -> None:
    print("Detalle por query (formato hit/mrr/P):\n")
    header = f"{'ID':<5}{'Query':<48}"
    for r in results:
        header += f"{r['system'][:11]:>15}"
    print(header)
    print("-" * len(header))
    n_queries = len(results[0]["per_query"])
    for i in range(n_queries):
        q = results[0]["per_query"][i]
        row = f"{q['id']:<5}{q['query'][:46]:<48}"
        for r in results:
            pq = r["per_query"][i]
            row += f"   {pq['hit_rate']:>1.0f}/{pq['mrr']:>4.2f}/{pq['precision']:>4.2f}"
        print(row)
    print()


def main() -> None:
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)["queries"]
    logger.info(f"Dataset cargado: {len(dataset)} queries")

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    hybrid = GrauRetriever(
        collection=collection,
        n_results=K,
        n_candidates=N_CANDIDATES,
    )
    dense = GrauRetriever(
        collection=collection,
        n_results=K,
        n_candidates=N_CANDIDATES,
        bm25_index_path="__force_dense_only__",
    )

    logger.info("Evaluando DENSE-ONLY...")
    res_dense = evaluate_system("dense_only", dense, dataset)
    logger.info("Evaluando HYBRID (dense + BM25 + RRF)...")
    res_hybrid = evaluate_system("hybrid", hybrid, dataset)

    print_summary([res_dense, res_hybrid])
    print_per_query([res_dense, res_hybrid])

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"k": K, "n_candidates": N_CANDIDATES, "results": [res_dense, res_hybrid]},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Resultados guardados en {RESULTS_PATH}")


if __name__ == "__main__":
    main()
