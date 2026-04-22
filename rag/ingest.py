from __future__ import annotations
import os
import re
import chess.pgn
import io
from typing import Generator
from contracts.partida import PartidaGrau, ChunkMetadata
from core.logging import get_logger

logger = get_logger(__name__)

TOMOS: dict[str, tuple[int, str]] = {
    "Grau I.pgn": (1, "Rudimentos"),
    "Grau II.pgn": (2, "Estrategia"),
    "Grau III.pgn": (3, "Medio juego / Conformación de peones"),
    "Grau IV.pgn": (4, "Estrategia Superior"),
}

MAX_CHARS_PER_CHUNK = 1800  # ~450 tokens; ventana pensada para análisis pedagógico
OVERLAP_SENTENCES = 1  # oraciones compartidas entre chunks adyacentes


def _extract_comments_from_node(node: chess.pgn.GameNode) -> str:
    """Extrae recursivamente todos los comentarios de un nodo y sus variantes."""
    parts: list[str] = []
    if node.comment:
        parts.append(node.comment.strip())
    for variation in node.variations:
        parts.extend(_collect_node_texts(variation))
    return " ".join(parts)


def _collect_node_texts(node: chess.pgn.GameNode) -> list[str]:
    texts: list[str] = []
    if node.comment:
        texts.append(node.comment.strip())
    for child in node.variations:
        texts.extend(_collect_node_texts(child))
    return texts


def _game_to_text(game: chess.pgn.Game) -> tuple[str, str]:
    """Devuelve (jugadas_str, comentarios_str) de una partida."""
    moves_parts: list[str] = []
    comments_parts: list[str] = []

    # Comentario antes de la primera jugada
    if game.comment:
        comments_parts.append(game.comment.strip())

    node = game
    while node.variations:
        next_node = node.variations[0]
        board = node.board()
        move = next_node.move
        san = board.san(move)
        move_num = board.fullmove_number
        turn = "w" if board.turn == chess.WHITE else "b"

        if turn == "w":
            moves_parts.append(f"{move_num}.{san}")
        else:
            moves_parts.append(san)

        if next_node.comment:
            comments_parts.append(next_node.comment.strip())

        # Variantes
        for var in node.variations[1:]:
            var_texts = _collect_node_texts(var)
            comments_parts.extend(var_texts)

        node = next_node

    return " ".join(moves_parts), " ".join(comments_parts)


def _build_texto_para_embedding(tomo: int, tema: str, eco: str, analisis_chunk: str) -> str:
    """Texto que va al embedding: contexto pedagógico mínimo + análisis de Grau.

    Se omite deliberadamente la notación algebraica y los nombres de jugadores:
    son ruido para búsquedas conceptuales.
    """
    parts = [f"Tomo {tomo}: {tema}"]
    if eco:
        parts.append(f"Apertura: {eco}")
    parts.append(analisis_chunk)
    return "\n".join(parts)


def _chunk_analisis(comentarios: str, max_chars: int, overlap_sentences: int) -> list[str]:
    """Divide el análisis pedagógico en ventanas por oración con solapamiento."""
    texto = comentarios.strip()
    if not texto:
        return []

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', texto) if s.strip()]
    if not sentences:
        return []

    # Si el total cabe en un chunk, no dividir
    total = sum(len(s) + 1 for s in sentences)
    if total <= max_chars:
        return [" ".join(sentences)]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current and current_len + len(sent) + 1 > max_chars:
            chunks.append(" ".join(current))
            tail = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(tail)
            current_len = sum(len(s) + 1 for s in current)
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def parse_pgn_file(filepath: str, tomo: int, tema: str) -> Generator[PartidaGrau, None, None]:
    logger.info(f"Parseando {filepath} (Tomo {tomo}: {tema})")
    count = 0
    skipped_sin_analisis = 0

    with open(filepath, encoding="utf-8-sig") as f:
        content = f.read()

    pgn_io = io.StringIO(content)

    while True:
        try:
            game = chess.pgn.read_game(pgn_io)
        except Exception as e:
            logger.warning(f"Error leyendo partida en {filepath}: {e}")
            continue

        if game is None:
            break

        headers = game.headers
        fen = headers.get("FEN", "")
        eco = headers.get("ECO", "")
        jugadas, comentarios = _game_to_text(game)

        analisis_chunks = _chunk_analisis(comentarios, MAX_CHARS_PER_CHUNK, OVERLAP_SENTENCES)

        if not analisis_chunks:
            skipped_sin_analisis += 1
            count += 1
            continue

        for chunk_idx, analisis in enumerate(analisis_chunks):
            partida_id = f"tomo{tomo}-{count}-chunk{chunk_idx}"
            texto_embedding = _build_texto_para_embedding(tomo, tema, eco, analisis)
            meta = ChunkMetadata(
                tomo=tomo,
                tema=tema,
                event=headers.get("Event", "?"),
                white=headers.get("White", "?"),
                black=headers.get("Black", "?"),
                result=headers.get("Result", "*"),
                eco=eco,
                annotator=headers.get("Annotator", ""),
                fen=fen,
                ply_count=int(headers.get("PlyCount", 0) or 0),
                chunk_index=chunk_idx,
                partida_id=partida_id,
            )
            yield PartidaGrau(
                partida_id=partida_id,
                texto_completo=texto_embedding,
                jugadas=jugadas,
                comentarios=analisis,
                metadata=meta,
            )

        count += 1

    logger.info(
        f"Tomo {tomo}: {count} partidas procesadas "
        f"({skipped_sin_analisis} descartadas por no tener análisis)"
    )


def ingest_all(data_dir: str) -> list[PartidaGrau]:
    all_chunks: list[PartidaGrau] = []
    for filename, (tomo, tema) in TOMOS.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Archivo no encontrado: {filepath}")
            continue
        for chunk in parse_pgn_file(filepath, tomo, tema):
            all_chunks.append(chunk)
    logger.info(f"Total chunks generados: {len(all_chunks)}")
    return all_chunks
