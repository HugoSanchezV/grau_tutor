"""Faithfulness eval del agente.

Para cada query del dataset, invoca al GrauAgent y mide:
- tool_use_rate: ¿el agente llamó a search_grau? (señal clave: NO alucina)
- citation_rate: ¿la respuesta cita una fuente del corpus? (regex Tomo/partida_id)
- groundedness: ¿menciona keywords del concepto del corpus?
- abstention_rate: para queries fuera de dominio, ¿se abstiene correctamente?

No usa LLM-as-judge — métricas deterministas, reproducibles, sin coste extra de tokens.
"""
from __future__ import annotations
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.react_agent import GrauAgent
from core.llm import get_llm
from core.logging import get_logger, setup_logging
from rag.retrieval import GrauRetriever
from rag.store import get_chroma_client, get_or_create_collection
from evals.metrics import aggregate

setup_logging()
logger = get_logger(__name__)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "faithfulness_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "faithfulness_results.json")

# Cita estricta: identifica el pasaje exacto del corpus (tomo, ECO, partida_id)
_STRICT_CITATION_RE = re.compile(
    r"(tomo\s*[1-4]|tomo\d-\d+|ECO\s*[A-E]\d{2}|partida\s+\d+)",
    re.IGNORECASE,
)
# Cita débil: solo nombra al autor o el libro, sin localizar el pasaje
_WEAK_CITATION_RE = re.compile(
    r"(grau|tratado general)",
    re.IGNORECASE,
)


def used_search_grau(reasoning: list[dict]) -> bool:
    return any(
        step.get("type") == "tool_call" and step.get("name") == "search_grau"
        for step in reasoning
    )


def has_strict_citation(reply: str) -> bool:
    return bool(_STRICT_CITATION_RE.search(reply))


def has_weak_citation(reply: str) -> bool:
    return bool(_WEAK_CITATION_RE.search(reply))


def keyword_overlap(reply: str, expected: list[str]) -> float:
    """Fracción de keywords esperados presentes en la respuesta (case-insensitive)."""
    if not expected:
        return 1.0
    text = reply.lower()
    hits = sum(1 for kw in expected if kw.lower() in text)
    return hits / len(expected)


def is_grounded(reply: str, expected: list[str], threshold: float = 0.25) -> bool:
    """La respuesta está fundamentada si menciona al menos `threshold` de los keywords."""
    return keyword_overlap(reply, expected) >= threshold


def evaluate_query(agent: GrauAgent, q: dict) -> dict:
    logger.info(f"[{q['id']}] {q['query'][:60]}")
    try:
        response = agent.chat(q["query"], thread_id=f"eval-{q['id']}")
        reply = response.reply
        reasoning = response.reasoning
        crashed = False
        error_msg = ""
    except Exception as e:
        logger.warning(f"[{q['id']}] crash durante invocación: {type(e).__name__}: {e}")
        reply = ""
        reasoning = []
        crashed = True
        error_msg = f"{type(e).__name__}: {str(e)[:200]}"

    used_search = used_search_grau(reasoning)
    cited_strict = has_strict_citation(reply)
    cited_weak = has_weak_citation(reply)
    cited_any = cited_strict or cited_weak
    grounded = is_grounded(reply, q["must_mention_any"])
    overlap = keyword_overlap(reply, q["must_mention_any"])

    # Para queries out_of_scope, "abstención" = NO usó search_grau Y mencionó al menos un keyword de "no puedo"
    is_out_of_scope = q.get("must_abstain", False)
    if is_out_of_scope:
        abstained = (not used_search) and grounded
        passed = abstained
    else:
        # In-scope: debe usar search_grau, citar fuente (cualquiera) y estar fundamentada
        checks = []
        if q.get("must_use_search", False):
            checks.append(used_search)
        if q.get("must_cite_source", False):
            checks.append(cited_any)
        checks.append(grounded)
        passed = all(checks)

    return {
        "id": q["id"],
        "tipo": q["tipo"],
        "query": q["query"],
        "reply_preview": reply[:200],
        "used_search_grau": used_search,
        "has_strict_citation": cited_strict,
        "has_weak_citation": cited_weak,
        "has_citation_any": cited_any,
        "keyword_overlap": round(overlap, 3),
        "grounded": grounded,
        "crashed": crashed,
        "error": error_msg,
        "passed": passed,
    }


def print_summary(results: list[dict]) -> None:
    in_scope = [r for r in results if r["tipo"] != "out_of_scope"]
    oos = [r for r in results if r["tipo"] == "out_of_scope"]

    print()
    print("=" * 60)
    print(f"FAITHFULNESS EVAL — {len(results)} queries")
    print("=" * 60)
    print(f"{'Métrica':<28} {'In-scope':>12} {'Out-of-scope':>15}")
    print("-" * 60)

    def _rate(items: list[dict], key: str) -> float:
        return aggregate([1.0 if r[key] else 0.0 for r in items])

    metrics = [
        ("tool_use (search_grau)", "used_search_grau"),
        ("citation strict (Tomo X)", "has_strict_citation"),
        ("citation weak (Grau)", "has_weak_citation"),
        ("citation any", "has_citation_any"),
        ("groundedness", "grounded"),
        ("crashed (model errors)", "crashed"),
        ("passed (overall)", "passed"),
    ]
    for label, key in metrics:
        in_v = _rate(in_scope, key) if in_scope else 0.0
        oos_v = _rate(oos, key) if oos else 0.0
        print(f"{label:<28} {in_v:>12.3f} {oos_v:>15.3f}")

    avg_overlap_in = aggregate([r["keyword_overlap"] for r in in_scope]) if in_scope else 0.0
    print(f"{'avg keyword_overlap':<28} {avg_overlap_in:>12.3f} {'-':>15}")
    print()


def print_detail(results: list[dict]) -> None:
    header = f"{'ID':<5}{'Tipo':<14}{'Search':>8}{'CiteStr':>9}{'CiteWk':>8}{'Overlap':>9}{'Grnd':>6}{'PASS':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['id']:<5}"
            f"{r['tipo']:<14}"
            f"{'Y' if r['used_search_grau'] else '-':>8}"
            f"{'Y' if r['has_strict_citation'] else '-':>9}"
            f"{'Y' if r['has_weak_citation'] else '-':>8}"
            f"{r['keyword_overlap']:>9.2f}"
            f"{'Y' if r['grounded'] else '-':>6}"
            f"{'PASS' if r['passed'] else 'FAIL':>6}"
        )
    print()


def main() -> None:
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)["queries"]
    logger.info(f"Dataset cargado: {len(dataset)} queries")

    client = get_chroma_client()
    collection = get_or_create_collection(client)
    retriever = GrauRetriever(collection)
    # Usamos OpenAI gpt-4o-mini para la eval: tool calling más fiable que Groq llama-3.3.
    # La métrica importante es si el sistema agéntico funciona, no el modelo concreto en producción.
    llm = get_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
    agent = GrauAgent(retriever=retriever, llm=llm, stateless=True)
    logger.info("Agente inicializado (stateless, openai/gpt-4o-mini)")

    results = [evaluate_query(agent, q) for q in dataset]
    print_summary(results)
    print_detail(results)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    logger.info(f"Resultados guardados en {RESULTS_PATH}")


if __name__ == "__main__":
    main()
