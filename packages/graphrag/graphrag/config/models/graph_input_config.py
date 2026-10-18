# Copyright (C) 2026 luke russo -- ruff complaining

"""Parameterization settings for the graph input section."""

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults


class GraphInputConfig(BaseModel):
    """The configuration section for a supplied graph.

    A graph-native run reads its graph from ``graph_input_storage`` instead of
    building one from documents.
    """

    entities_file: str = Field(
        description="The file name of the supplied entities table.",
        default=graphrag_config_defaults.graph_input.entities_file,
    )

    relationships_file: str = Field(
        description="The file name of the supplied relationships table.",
        default=graphrag_config_defaults.graph_input.relationships_file,
    )

    text_units_file: str = Field(
        description="The file name of the supplied text units table.",
        default=graphrag_config_defaults.graph_input.text_units_file,
    )
