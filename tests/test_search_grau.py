"""Tests de la herramienta search_grau (mockea el retriever para no depender de ChromaDB)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock
import pytest

from agents.tools.search_grau import (
    ChunkResult,
    SearchGrauInput,
    search_grau,
    format_chunks_for_llm,
    build_search_grau_tool,
)


def _make_item(
    partida_id: str,
    tomo: int = 2,
    tema: str = "Estrategia",
    white: str = "Capablanca",
    black: str = "Lasker",
    eco: str = "C60",
    fen: str = "",
    jugadas: str = "1.e4 e5",
    doc: str = "Texto pedagógico de Grau sobre la clavada.",
    similarity: float = 0.82,
) -> dict:
    return {
        "id": partida_id,
        "doc": doc,
        "meta": {
            "partida_id": partida_id,
            "tomo": tomo,
            "tema": tema,
            "white": white,
            "black": black,
            "result": "1-0",
            "eco": eco,
            "fen": fen,
            "jugadas": jugadas,
        },
        "similarity": similarity,
    }


def _retriever_with(items: list[dict]) -> MagicMock:
    """Devuelve un retriever mockeado que siempre responde con `items`."""
    mock = MagicMock()
    mock.retrieve_raw.return_value = items
    return mock


# ---------- ChunkResult ----------


def test_chunk_result_from_retrieval_item():
    item = _make_item("tomo2-5")
    chunk = ChunkResult.from_retrieval_item(item)
    assert chunk.partida_id == "tomo2-5"
    assert chunk.tomo == 2
    assert chunk.tema == "Estrategia"
    assert chunk.white == "Capablanca"
    assert chunk.eco == "C60"
    assert chunk.similarity == pytest.approx(0.82)
    assert "clavada" in chunk.texto


def test_chunk_result_sin_similarity():
    item = _make_item("tomo1-1", similarity=None)
    item["similarity"] = None
    chunk = ChunkResult.from_retrieval_item(item)
    assert chunk.similarity is None


# ---------- search_grau: top-k y filtros ----------


def test_search_grau_respeta_k():
    items = [_make_item(f"tomo2-{i}") for i in range(10)]
    retriever = _retriever_with(items)
    out = search_grau(retriever, query="clavada", k=3)
    assert len(out) == 3
    assert all(isinstance(c, ChunkResult) for c in out)
    retriever.retrieve_raw.assert_called_once()


def test_search_grau_filtra_por_tomo():
    items = [
        _make_item("tomo1-1", tomo=1, tema="Rudimentos"),
        _make_item("tomo2-1", tomo=2, tema="Estrategia"),
        _make_item("tomo2-2", tomo=2, tema="Estrategia"),
        _make_item("tomo3-1", tomo=3, tema="Medio juego"),
    ]
    retriever = _retriever_with(items)
    out = search_grau(retriever, query="peones", k=5, tomo=2)
    assert len(out) == 2
    assert all(c.tomo == 2 for c in out)


def test_search_grau_filtra_por_tema_case_insensitive():
    items = [
        _make_item("tomo1-1", tomo=1, tema="Rudimentos"),
        _make_item("tomo2-1", tomo=2, tema="Estrategia"),
    ]
    retriever = _retriever_with(items)
    out = search_grau(retriever, query="x", k=5, tema="estrategia")
    assert len(out) == 1
    assert out[0].tomo == 2


def test_search_grau_con_filtros_amplia_el_pool():
    """Cuando hay filtros, debe pedir más candidatos al retriever (pool = k * 4)."""
    items = [_make_item(f"tomo2-{i}", tomo=2) for i in range(20)]
    retriever = _retriever_with(items)
    search_grau(retriever, query="x", k=5, tomo=2)
    call_kwargs = retriever.retrieve_raw.call_args.kwargs
    assert call_kwargs.get("n_results") == 20


def test_search_grau_sin_filtros_no_amplia_el_pool():
    items = [_make_item(f"tomo2-{i}") for i in range(5)]
    retriever = _retriever_with(items)
    search_grau(retriever, query="x", k=5)
    call_kwargs = retriever.retrieve_raw.call_args.kwargs
    assert call_kwargs.get("n_results") == 5


def test_search_grau_devuelve_vacio_si_filtro_no_matchea():
    items = [_make_item("tomo2-1", tomo=2)]
    retriever = _retriever_with(items)
    out = search_grau(retriever, query="x", k=5, tomo=4)
    assert out == []


# ---------- format_chunks_for_llm ----------


def test_format_sin_resultados():
    out = format_chunks_for_llm([])
    assert "Sin resultados" in out


def test_format_incluye_metadata_clave():
    items = [_make_item("tomo2-1", fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")]
    chunks = [ChunkResult.from_retrieval_item(items[0])]
    out = format_chunks_for_llm(chunks)
    assert "Fuente 1" in out
    assert "Tomo 2" in out
    assert "Capablanca" in out
    assert "ECO C60" in out
    assert "id=tomo2-1" in out
    assert "FEN:" in out
    assert "Jugadas:" in out
    assert "sim=0.82" in out


def test_format_sin_fen_no_muestra_linea_fen():
    items = [_make_item("tomo2-1", fen="")]
    chunks = [ChunkResult.from_retrieval_item(items[0])]
    out = format_chunks_for_llm(chunks)
    assert "FEN:" not in out


# ---------- SearchGrauInput validaciones ----------


def test_input_schema_valida_rangos():
    SearchGrauInput(query="clavada", k=5, tomo=2)  # ok
    with pytest.raises(Exception):
        SearchGrauInput(query="x", k=0)
    with pytest.raises(Exception):
        SearchGrauInput(query="x", k=16)
    with pytest.raises(Exception):
        SearchGrauInput(query="x", tomo=5)
    with pytest.raises(Exception):
        SearchGrauInput(query="x", tomo=0)


# ---------- build_search_grau_tool ----------


def test_build_tool_metadata():
    retriever = _retriever_with([])
    tool = build_search_grau_tool(retriever)
    assert tool.name == "search_grau"
    assert "Grau" in tool.description
    assert tool.args_schema is SearchGrauInput


def test_build_tool_invocacion_devuelve_string():
    items = [_make_item("tomo2-1"), _make_item("tomo2-2")]
    retriever = _retriever_with(items)
    tool = build_search_grau_tool(retriever)
    result = tool.invoke({"query": "clavada", "k": 2})
    assert isinstance(result, str)
    assert "Fuente 1" in result
    assert "Fuente 2" in result


def test_build_tool_pasa_filtros_al_retriever():
    items = [_make_item("tomo2-1", tomo=2), _make_item("tomo4-1", tomo=4)]
    retriever = _retriever_with(items)
    tool = build_search_grau_tool(retriever)
    result = tool.invoke({"query": "x", "k": 5, "tomo": 4})
    assert "tomo4-1" in result
    assert "tomo2-1" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
