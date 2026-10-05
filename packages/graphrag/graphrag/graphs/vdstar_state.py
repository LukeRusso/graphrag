# Copyright (C) 2026 luke russo -- ruff complaining

import json

from graphrag_storage import Storage

GRAPH_BLOB_KEY = "graph_state"
"""Storage key for the C++ ``GraphStore`` blob (raw bytes)."""

ID_MAP_KEY = "node_id_map"
"""Storage key for the ``str -> int`` node id map (JSON)."""


class VDStarState:
    """Read and write the two-key VD-STAR state pair through ``Storage``.

    The pair is a C++ ``GraphStore`` blob plus the ``str -> int`` node id map (mapping from graphrag node id to VD-STAR node id).
    """

    def __init__(self, storage: Storage) -> None:
        """Wrap the storage namespace holding the state pair."""
        self._storage = storage

    async def load(self) -> tuple[bytes, dict[str, int]] | None:
        """Load ``(blob, id_map)``, or ``None`` when both keys are absent.

        Raises ``RuntimeError`` when exactly one key is present, which means a
        write was interrupted between the two keys.
        """
        has_blob = await self._storage.has(GRAPH_BLOB_KEY)
        has_id_map = await self._storage.has(ID_MAP_KEY)

        if not has_blob and not has_id_map:
            return None

        if has_blob != has_id_map:
            present = GRAPH_BLOB_KEY if has_blob else ID_MAP_KEY
            missing = ID_MAP_KEY if has_blob else GRAPH_BLOB_KEY
            msg = (
                f"Lopsided VD-STAR state: '{present}' is present but "
                f"'{missing}' is not. This indicates an interrupted write "
                "between the two state keys. Rebuild the state from fresh."
            )
            raise RuntimeError(msg)

        blob = await self._storage.get(GRAPH_BLOB_KEY, as_bytes=True)
        raw_id_map = await self._storage.get(ID_MAP_KEY, as_bytes=True)

        return blob, self.decode_id_map(raw_id_map)

    async def save(self, blob: bytes, id_map: dict[str, int]) -> None:
        """Persist ``(blob, id_map)``."""
        await self._storage.set(GRAPH_BLOB_KEY, blob)
        await self._storage.set(ID_MAP_KEY, json.dumps(id_map))

    def decode_id_map(self, raw: bytes | str) -> dict[str, int]:
        """Decode the persisted node id map, rejecting non-int values.

        ``bool`` passes, since ``isinstance(True, int)`` is ``True`` and ``True == 1``.
        """
        decoded = json.loads(raw)

        if not isinstance(decoded, dict):
            msg = f"Node id map must be a JSON object, got {type(decoded).__name__}."
            raise TypeError(msg)

        id_map: dict[str, int] = {}
        for key, value in decoded.items():
            if not isinstance(value, int):
                msg = f"Node id for '{key}' is {type(value).__name__}, expected int."
                raise TypeError(msg)
            id_map[key] = value

        return id_map
