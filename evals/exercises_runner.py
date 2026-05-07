"""Eval del motor de ejercicios: generate_exercise + evaluate_answer.

Mide tres dimensiones independientes:

1. **Validez de generación** — ¿`generate_exercise(tema)` produce ejercicios bien formados?
   Métricas: validez_rate (FEN parseable), expected_move_rate (jugada correcta no None).

2. **Precisión de evaluación (ground truth)** — dado un FEN del corpus + jugada esperada,
   ¿`evaluate_answer(jugada_esperada)` la marca como correcta?
   Métrica: accuracy_correcta.

3. **Detección de jugadas ilegales** — ¿`evaluate_answer` rechaza correctamente jugadas inválidas?
   Métrica: accuracy_ilegal.

4. **Detección de alternativas tácticas** — para cada FEN, probamos una jugada legal pero
   distinta a la esperada y verificamos que el flag `alternativa_valida` sea coherente con
   la diferencia de fortaleza táctica.
"""
from __future__ import annotations
import json
import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.tools.exercise_gen import evaluate_answer, generate_exercise
from core.logging import get_logger, setup_logging
from rag.retrieval import GrauRetriever
from rag.store import get_chroma_client, get_or_create_collection
from evals.metrics import aggregate

setup_logging()
logger = get_logger(__name__)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "exercises_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "exercises_results.json")


def _pick_alternative_legal_move(fen: str, expected: str) -> str | None:
    """Devuelve un movimiento legal del FEN distinto al esperado.

    Estrategia: tomar la primera jugada legal que no coincida con `expected` en SAN.
    Útil para probar evaluate_answer con jugadas alternativas.
    """
    board = chess.Board(fen)
    try:
        expected_move = board.parse_san(expected)
        expected_san = board.san(expected_move)
    except Exception:
        expected_san = expected
    for move in board.legal_moves:
        san = board.san(move)
        if san != expected_san:
            return san
    return None


def evaluate_generation(retriever: GrauRetriever, themes: list[str]) -> list[dict]:
    """Para cada tema, genera un ejercicio y verifica validez."""
    results = []
    for tema in themes:
        ej = generate_exercise(retriever, tema=tema)
        if ej is None:
            results.append({
                "tema": tema,
                "generated": False,
                "fen_valid": False,
                "has_expected_move": False,
                "comentario_present": False,
            })
            continue
        fen_valid = False
        try:
            chess.Board(ej.fen)
            fen_valid = True
        except Exception:
            pass
        results.append({
            "tema": tema,
            "generated": True,
            "fen_valid": fen_valid,
            "has_expected_move": ej.jugada_correcta is not None,
            "comentario_present": bool(ej.comentario_grau and len(ej.comentario_grau) > 30),
            "fen": ej.fen,
            "expected_move": ej.jugada_correcta,
            "partida_id": ej.partida_id,
            "tomo": ej.tomo,
        })
    return results


def evaluate_correctness(exercises: list[dict]) -> list[dict]:
    """Para cada ejercicio, verifica:
    - evaluate_answer(expected, expected) → correcta=True
    - evaluate_answer("Xx99", expected) → legal=False
    - evaluate_answer(alternativa, expected) → flag alternativa coherente
    """
    results = []
    for ex in exercises:
        fen = ex["fen"]
        expected = ex["expected_move"]

        # 1. Jugada correcta
        ev_correct = evaluate_answer(fen, expected, expected)
        marca_correcta = bool(ev_correct.correcta) and ev_correct.legal

        # 2. Jugada ilegal sintácticamente
        ev_illegal = evaluate_answer(fen, "Zz99", expected)
        detecta_ilegal = (not ev_illegal.legal) and ev_illegal.correcta is False

        # 3. Alternativa legal
        alt = _pick_alternative_legal_move(fen, expected)
        if alt:
            ev_alt = evaluate_answer(fen, alt, expected)
            alt_legal = ev_alt.legal
            alt_marked_alternative = ev_alt.alternativa_valida
        else:
            alt_legal = False
            alt_marked_alternative = False

        results.append({
            "id": ex["id"],
            "partida_id": ex["partida_id"],
            "tipo": ex["tipo"],
            "expected_move": expected,
            "marca_correcta": marca_correcta,
            "detecta_ilegal": detecta_ilegal,
            "alt_move": alt,
            "alt_legal": alt_legal,
            "alt_alternativa_valida": alt_marked_alternative,
        })
    return results


def print_generation_summary(results: list[dict]) -> None:
    n = len(results)
    if n == 0:
        return
    gen_rate = aggregate([1.0 if r["generated"] else 0.0 for r in results])
    valid_rate = aggregate([1.0 if r["fen_valid"] else 0.0 for r in results])
    move_rate = aggregate([1.0 if r["has_expected_move"] else 0.0 for r in results])
    com_rate = aggregate([1.0 if r["comentario_present"] else 0.0 for r in results])

    print()
    print("=" * 60)
    print(f"GENERATE_EXERCISE — {n} temas")
    print("=" * 60)
    print(f"  generated_rate          {gen_rate:.3f}")
    print(f"  fen_valid_rate          {valid_rate:.3f}")
    print(f"  expected_move_rate      {move_rate:.3f}")
    print(f"  comentario_present_rate {com_rate:.3f}")
    print()
    for r in results:
        status = "OK" if r["generated"] and r["fen_valid"] else "FAIL"
        line = f"  [{status}] tema={r['tema']!r:<22}"
        if r["generated"]:
            line += f" partida={r.get('partida_id', '?')} expected={r.get('expected_move', '?')}"
        print(line)
    print()


def print_correctness_summary(results: list[dict]) -> None:
    n = len(results)
    if n == 0:
        return
    correct_rate = aggregate([1.0 if r["marca_correcta"] else 0.0 for r in results])
    illegal_rate = aggregate([1.0 if r["detecta_ilegal"] else 0.0 for r in results])

    print("=" * 60)
    print(f"EVALUATE_ANSWER — {n} ejercicios")
    print("=" * 60)
    print(f"  marca correcta cuando jugada=esperada    {correct_rate:.3f}")
    print(f"  detecta jugada ilegal                    {illegal_rate:.3f}")
    print()
    print(f"{'ID':<6}{'Partida':<14}{'Esperada':>10}{'Marca OK':>10}{'Ilegal OK':>11}{'Alt SAN':>10}{'Alt valida':>12}")
    print("-" * 73)
    for r in results:
        print(
            f"{r['id']:<6}"
            f"{r['partida_id']:<14}"
            f"{r['expected_move']:>10}"
            f"{'Y' if r['marca_correcta'] else '-':>10}"
            f"{'Y' if r['detecta_ilegal'] else '-':>11}"
            f"{(r['alt_move'] or '-'):>10}"
            f"{'Y' if r['alt_alternativa_valida'] else '-':>12}"
        )
    print()


def main() -> None:
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    exercises = data["exercises"]
    themes = data["themes_for_generation"]

    client = get_chroma_client()
    collection = get_or_create_collection(client)
    retriever = GrauRetriever(collection)

    logger.info(f"Evaluando generate_exercise sobre {len(themes)} temas...")
    gen_results = evaluate_generation(retriever, themes)
    print_generation_summary(gen_results)

    logger.info(f"Evaluando evaluate_answer sobre {len(exercises)} ejercicios...")
    ev_results = evaluate_correctness(exercises)
    print_correctness_summary(ev_results)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"generation": gen_results, "correctness": ev_results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Resultados guardados en {RESULTS_PATH}")


if __name__ == "__main__":
    main()
