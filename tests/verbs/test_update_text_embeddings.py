# Copyright (C) 2026 Microsoft
# Licensed under the MIT License

"""Verb test for the update_text_embeddings workflow."""

from unittest.mock import patch

from graphrag.config.embeddings import all_embeddings
from graphrag.index.workflows.update_text_embeddings import (
    run_workflow,
)
from graphrag_vectors import VectorStoreDocument, create_vector_store

from tests.unit.config.utils import get_default_graphrag_config

from .util import create_test_context


def _config(tmp_path):
    config = get_default_graphrag_config()
    llm_settings = config.get_embedding_model_config(
        config.embed_text.embedding_model_id
    )
    llm_settings.type = "mock"
    llm_settings.mock_responses = [1.0] * 3072

    config.embed_text.names = list(all_embeddings)
    config.snapshots.embeddings = True
    config.vector_store.db_uri = str(tmp_path / "lancedb")
    return config


def _store(config):
    store = create_vector_store(
        config.vector_store, config.vector_store.index_schema["community_full_content"]
    )
    store.connect()
    return store


async def _run(config, context):
    context.state["update_timestamp"] = "20260220-000000"
    with patch(
        "graphrag.index.workflows.update_text_embeddings.get_update_table_providers",
    ) as mock_providers:
        mock_providers.return_value = (
            context.output_table_provider,
            None,
            None,
        )
        await run_workflow(config, context)


async def test_update_text_embeddings(tmp_path):
    """Verify update_text_embeddings produces embedding tables."""
    context = await create_test_context(
        storage=[
            "documents",
            "relationships",
            "text_units",
            "entities",
            "community_reports",
        ]
    )
    config = _config(tmp_path)

    await _run(config, context)

    parquet_files = context.output_storage.keys()
    for field in all_embeddings:
        assert f"embeddings.{field}.parquet" in parquet_files

    entity_embeddings = await context.output_table_provider.read_dataframe(
        "embeddings.entity_description"
    )
    assert len(entity_embeddings.columns) == 2
    assert "id" in entity_embeddings.columns
    assert "embedding" in entity_embeddings.columns

    reports = await context.output_table_provider.read_dataframe("community_reports")
    assert set(_store(config).ids()) == set(reports["id"].astype(str))


async def test_an_unchanged_row_is_not_embedded_again(tmp_path):
    context = await create_test_context(storage=["community_reports", "entities"])
    config = _config(tmp_path)
    await _run(config, context)
    stored = sorted(_store(config).ids())

    await _run(config, context)

    assert sorted(_store(config).ids()) == stored


async def test_a_row_the_table_no_longer_holds_is_removed(tmp_path):
    context = await create_test_context(storage=["community_reports"])
    config = _config(tmp_path)
    await _run(config, context)
    store = _store(config)
    store.insert(VectorStoreDocument(id="stale-row", vector=[0.0] * 3072))
    assert "stale-row" in store.ids()

    await _run(config, context)

    assert "stale-row" not in _store(config).ids()
