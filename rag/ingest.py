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

MAX_TOKENS_PER_CHUNK = 1800  # ~7200 chars aprox; split si supera esto


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


def _build_texto_completo(game: chess.pgn.Game, jugadas: str, comentarios: str) -> str:
    headers = game.headers
    parts = []

    event = headers.get("Event", "?")
    white = headers.get("White", "?")
    black = headers.get("Black", "?")
    eco = headers.get("ECO", "")
    result = headers.get("Result", "*")

    parts.append(f"Partida: {white} vs {black}")
    if eco:
        parts.append(f"Apertura ECO: {eco}")
    parts.append(f"Resultado: {result}")
    if jugadas:
        parts.append(f"Jugadas: {jugadas}")
    if comentarios:
        parts.append(f"Análisis de Grau: {comentarios}")

    return "\n".join(parts)


def _split_texto(texto: str, max_chars: int = 7200) -> list[str]:
    """Divide texto largo en chunks manteniendo párrafos completos."""
    if len(texto) <= max_chars:
        return [texto]

    sentences = re.split(r'(?<=[.!?])\s+', texto)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = s
        else:
            current = (current + " " + s).strip()
    if current:
        chunks.append(current.strip())
    return chunks if chunks else [texto[:max_chars]]


def parse_pgn_file(filepath: str, tomo: int, tema: str) -> Generator[PartidaGrau, None, None]:
    logger.info(f"Parseando {filepath} (Tomo {tomo}: {tema})")
    count = 0

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
        jugadas, comentarios = _game_to_text(game)
        texto = _build_texto_completo(game, jugadas, comentarios)
        subtextos = _split_texto(texto)

        for chunk_idx, subtexto in enumerate(subtextos):
            partida_id = f"tomo{tomo}-{count}-chunk{chunk_idx}"
            meta = ChunkMetadata(
                tomo=tomo,
                tema=tema,
                event=headers.get("Event", "?"),
                white=headers.get("White", "?"),
                black=headers.get("Black", "?"),
                result=headers.get("Result", "*"),
                eco=headers.get("ECO", ""),
                annotator=headers.get("Annotator", ""),
                fen=fen,
                ply_count=int(headers.get("PlyCount", 0) or 0),
                chunk_index=chunk_idx,
                partida_id=partida_id,
            )
            yield PartidaGrau(
                partida_id=partida_id,
                texto_completo=subtexto,
                jugadas=jugadas,
                comentarios=comentarios,
                metadata=meta,
            )

        count += 1

    logger.info(f"Tomo {tomo}: {count} partidas procesadas")


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
