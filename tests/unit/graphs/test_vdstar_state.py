# Copyright (C) 2026 luke russo -- ruff complaining

import pytest
from graphrag.graphs.vdstar_state import (
    GRAPH_BLOB_KEY,
    ID_MAP_KEY,
    VDStarState,
)
from graphrag_storage.memory_storage import MemoryStorage


def _state() -> VDStarState:
    return VDStarState(MemoryStorage())


async def test_absent_state_returns_none():
    state = _state()

    assert await state.load() is None


async def test_save_then_load_round_trips():
    state = _state()
    blob = b"\x00\x01binary-blob\xff"
    id_map = {"node-a": 1, "node-b": 2, "node-c": 7}

    await state.save(blob, id_map)

    loaded = await state.load()

    assert loaded is not None
    loaded_blob, loaded_id_map = loaded
    assert loaded_blob == blob
    assert loaded_id_map == id_map


async def test_has_is_false_before_save():
    assert await _state().has() is False


async def test_has_is_true_after_save():
    state = _state()
    await state.save(b"blob", {"node-a": 1})

    assert await state.has() is True


@pytest.mark.parametrize(
    "present_key",
    [GRAPH_BLOB_KEY, ID_MAP_KEY],
    ids=["blob-only", "id-map-only"],
)
async def test_lopsided_pair_raises(present_key: str):
    storage = MemoryStorage()
    await storage.set(present_key, b"whatever")

    with pytest.raises(RuntimeError, match="Lopsided VD-STAR state"):
        await VDStarState(storage).load()


@pytest.mark.parametrize(
    "present_key",
    [GRAPH_BLOB_KEY, ID_MAP_KEY],
    ids=["blob-only", "id-map-only"],
)
async def test_has_is_false_for_a_lopsided_pair(present_key: str):
    storage = MemoryStorage()
    await storage.set(present_key, b"whatever")

    assert await VDStarState(storage).has() is False


async def test_decode_accepts_json_bytes():
    storage = MemoryStorage()
    await storage.set(GRAPH_BLOB_KEY, b"blob")
    await storage.set(ID_MAP_KEY, b'{"a": 1, "b": 2}')

    loaded = await VDStarState(storage).load()

    assert loaded is not None
    assert loaded[1] == {"a": 1, "b": 2}


def test_decode_rejects_non_integer_values():
    with pytest.raises(TypeError, match=r"Node id for 'a' is str, expected int"):
        _state().decode_id_map(b'{"a": "not-an-int"}')

    with pytest.raises(TypeError, match=r"Node id for 'a' is float, expected int"):
        _state().decode_id_map(b'{"a": 1.5}')


def test_decode_rejects_non_object_payload():
    with pytest.raises(TypeError, match="must be a JSON object"):
        _state().decode_id_map(b"[1, 2, 3]")
