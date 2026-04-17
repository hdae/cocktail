from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from cocktail_server.schemas.images import GeneratedImageRef
from cocktail_server.schemas.messages import Message


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class Session:
    """1 会話ぶんの内部状態。メッセージ列に加え、seed 継続とギャラリー用のメタを保持。

    `last_image_seed` は `seed_action="keep"` が参照する「直前に生成に使った seed」。
    `generated_images` は発行順（append 順 = 昇順）。ギャラリー横断ビューでは
    `list_all_generated_images` が全 Session を舐めて降順に並び替える。
    """

    id: str
    created_at: datetime
    updated_at: datetime
    messages: list[Message] = field(default_factory=list)
    last_image_seed: int | None = None
    generated_images: list[GeneratedImageRef] = field(default_factory=list)


class ConversationStore:
    """会話とメッセージ列のインメモリストア。

    M1 では永続化しない。M3 で SQLite 版に差し替える想定のため async な API にしている
    （in-memory 実装では実質同期だが、呼び出し側を将来変えずに済むようにするため）。
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def create(self) -> str:
        """新規会話 id を払い出す。空の履歴が作成される。"""
        conversation_id = str(uuid.uuid4())
        now = _utcnow()
        self._sessions[conversation_id] = Session(
            id=conversation_id,
            created_at=now,
            updated_at=now,
        )
        return conversation_id

    async def exists(self, conversation_id: str) -> bool:
        return conversation_id in self._sessions

    async def append(self, conversation_id: str, message: Message) -> None:
        """メッセージを末尾に追加する。`conversation_id` の整合性は呼び出し側で担保。"""
        session = self._sessions.get(conversation_id)
        if session is None:
            raise KeyError(conversation_id)
        if message.conversation_id != conversation_id:
            raise ValueError(
                f"message.conversation_id {message.conversation_id!r} != {conversation_id!r}"
            )
        session.messages.append(message)
        session.updated_at = _utcnow()

    async def list_messages(self, conversation_id: str) -> list[Message]:
        """履歴のスナップショットを返す（呼び出し側の変更がストアに波及しないようコピー）。"""
        session = self._sessions.get(conversation_id)
        if session is None:
            raise KeyError(conversation_id)
        return list(session.messages)

    async def record_generated_image(self, conversation_id: str, ref: GeneratedImageRef) -> None:
        """画像生成完了時に呼ぶ。ref.seed を last_image_seed に反映し、一覧にも追加する。"""
        session = self._sessions.get(conversation_id)
        if session is None:
            raise KeyError(conversation_id)
        if ref.conversation_id != conversation_id:
            raise ValueError(f"ref.conversation_id {ref.conversation_id!r} != {conversation_id!r}")
        session.generated_images.append(ref)
        session.last_image_seed = ref.seed
        session.updated_at = _utcnow()

    async def get_last_image_seed(self, conversation_id: str) -> int | None:
        """`seed_action="keep"` 解決用。会話が無い / 画像未生成なら None。"""
        session = self._sessions.get(conversation_id)
        if session is None:
            return None
        return session.last_image_seed

    async def list_all_generated_images(
        self, *, limit: int, before: datetime | None = None
    ) -> tuple[list[GeneratedImageRef], datetime | None]:
        """全 Session を横断した生成画像一覧を `created_at` 降順で返す。

        `before` が指定されれば `created_at < before` のみ。`limit+1` 件取って
        次ページがあるかを判定し、あれば `next_before = 最後の 1 件の created_at`。
        """
        all_refs: list[GeneratedImageRef] = []
        for session in self._sessions.values():
            all_refs.extend(session.generated_images)
        all_refs.sort(key=lambda r: r.created_at, reverse=True)
        if before is not None:
            all_refs = [r for r in all_refs if r.created_at < before]
        page = all_refs[: limit + 1]
        if len(page) > limit:
            return page[:limit], page[limit - 1].created_at
        return page, None
