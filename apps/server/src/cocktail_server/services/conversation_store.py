from __future__ import annotations

import uuid

from cocktail_server.schemas.messages import Message


class ConversationStore:
    """会話とメッセージ列のインメモリストア。

    M1 では永続化しない。M2 で SQLite 版に差し替える想定のため async な API にしている
    （in-memory 実装では実質同期だが、呼び出し側を将来変えずに済むようにするため）。
    """

    def __init__(self) -> None:
        self._messages: dict[str, list[Message]] = {}

    async def create(self) -> str:
        """新規会話 id を払い出す。空の履歴が作成される。"""
        conversation_id = str(uuid.uuid4())
        self._messages[conversation_id] = []
        return conversation_id

    async def exists(self, conversation_id: str) -> bool:
        return conversation_id in self._messages

    async def append(self, conversation_id: str, message: Message) -> None:
        """メッセージを末尾に追加する。`conversation_id` の整合性は呼び出し側で担保。"""
        if conversation_id not in self._messages:
            raise KeyError(conversation_id)
        if message.conversation_id != conversation_id:
            raise ValueError(
                f"message.conversation_id {message.conversation_id!r} != {conversation_id!r}"
            )
        self._messages[conversation_id].append(message)

    async def list_messages(self, conversation_id: str) -> list[Message]:
        """履歴のスナップショットを返す（呼び出し側の変更がストアに波及しないようコピー）。"""
        if conversation_id not in self._messages:
            raise KeyError(conversation_id)
        return list(self._messages[conversation_id])
