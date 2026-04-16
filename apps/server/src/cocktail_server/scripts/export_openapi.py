"""OpenAPI スキーマを `packages/api-types/openapi.json` に書き出す。

Makefile の `gen-types` から呼ばれ、続いて TS 側が openapi-typescript を回す想定。
SSE で返すイベント型は FastAPI の自動スキーマに載らないので、明示的に
`components.schemas` に注入して Zod と二重化できるようにしておく。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from cocktail_server.main import create_app
from cocktail_server.schemas.events import (
    AssistantEndEvent,
    AssistantStartEvent,
    ConversationEvent,
    DoneEvent,
    ErrorEvent,
    ImageReadyEvent,
    TextDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    UserSavedEvent,
)

SSE_EVENT_MODELS: tuple[type[BaseModel], ...] = (
    ConversationEvent,
    UserSavedEvent,
    AssistantStartEvent,
    TextDeltaEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ImageReadyEvent,
    AssistantEndEvent,
    ErrorEvent,
    DoneEvent,
)


def _inject_sse_event_schemas(schema: dict[str, Any]) -> None:
    _, definitions = models_json_schema(
        [(m, "validation") for m in SSE_EVENT_MODELS],
        ref_template="#/components/schemas/{model}",
    )
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    for name, body in definitions.get("$defs", {}).items():
        components.setdefault(name, body)


def build_schema() -> dict[str, Any]:
    app = create_app()
    schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )
    _inject_sse_event_schemas(schema)
    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OpenAPI schema")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("packages/api-types/openapi.json"),
        help="Output path (default: packages/api-types/openapi.json)",
    )
    args = parser.parse_args()

    schema = build_schema()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
