from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from cocktail_server.schemas.conversations import ConversationDetail, ConversationSummary
from cocktail_server.services.conversation_store import ConversationStore

router = APIRouter()


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(request: Request) -> list[ConversationSummary]:
    """全会話のサマリを `updated_at` 降順で返す。左メニューの履歴一覧向け。

    共有空間前提なので認可はしない。ページングは当面なし（M3 の SQLite 化時に再検討）。
    """
    store: ConversationStore = request.app.state.conversations
    return await store.list_conversations()


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str, request: Request) -> ConversationDetail:
    """会話 1 件の詳細（messages + 生成画像メタ）を返す。

    URL 直開き / リロード時に履歴を復元するためのエンドポイント。
    未知の id は 404。共有空間前提なので認可はしない。
    """
    store: ConversationStore = request.app.state.conversations
    try:
        session = await store.get_session(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc

    return ConversationDetail(
        id=session.id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=session.messages,
        generated_images=session.generated_images,
    )
