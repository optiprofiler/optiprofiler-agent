"""Decode-time JSON Schema constraints for Interpreter reports.

`BenchmarkReport` is the first constrained-decoding target because it is
closed, typed JSON. API-only providers keep the ordinary structured-output
path; a self-hosted vLLM OpenAI-compatible endpoint can opt into this
backend through ``LLMConfig(constrained_decoding=True)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel


class ReportConstraintBackend(Protocol):
    """Bind an LLM so report generation is constrained at decode time."""

    name: str

    def bind(self, llm: Any, schema: type[BaseModel]) -> Any:
        ...


@dataclass(frozen=True)
class VLLMJSONSchemaBackend:
    """Bind vLLM's OpenAI-compatible JSON Schema structured-output hint."""

    name: str = "vllm-json-schema"

    def bind(self, llm: Any, schema: type[BaseModel]) -> Any:
        return llm.bind(
            extra_body={
                "structured_outputs": {
                    "json": schema.model_json_schema(),
                },
            },
        )


__all__ = [
    "ReportConstraintBackend",
    "VLLMJSONSchemaBackend",
]
