"""Tests del pipeline de ingesta PGN."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from rag.ingest import parse_pgn_file, ingest_all
from contracts.partida import PartidaGrau


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def test_parse_tomo_i():
    chunks = list(parse_pgn_file(os.path.join(DATA_DIR, "Grau I.pgn"), 1, "Rudimentos"))
    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, PartidaGrau)
        assert c.metadata.tomo == 1
        assert c.metadata.tema == "Rudimentos"
        assert len(c.texto_completo) > 0
        assert c.partida_id.startswith("tomo1-")


def test_encoding_correcta():
    chunks = list(parse_pgn_file(os.path.join(DATA_DIR, "Grau I.pgn"), 1, "Rudimentos"))
    # Verificar que los acentos son correctos (ó = U+00F3, no bytes UTF-8 mal decodificados)
    textos_con_acento = [c for c in chunks if "ó" in c.texto_completo or "á" in c.texto_completo]
    assert len(textos_con_acento) > 0


def test_ingest_all_todos_los_tomos():
    chunks = ingest_all(DATA_DIR)
    assert len(chunks) > 600  # 1072 partidas, algunas subdivididas

    tomos = {c.metadata.tomo for c in chunks}
    assert tomos == {1, 2, 3, 4}


def test_to_chroma_document():
    chunks = list(parse_pgn_file(os.path.join(DATA_DIR, "Grau I.pgn"), 1, "Rudimentos"))
    doc = chunks[0].to_chroma_document()
    assert "id" in doc
    assert "document" in doc
    assert "metadata" in doc
    assert doc["metadata"]["tomo"] == 1


def test_ids_unicos():
    chunks = ingest_all(DATA_DIR)
    ids = [c.partida_id for c in chunks]
    assert len(ids) == len(set(ids)), "IDs duplicados encontrados"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
