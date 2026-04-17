from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field

from cocktail_server.schemas.events import SseEvent

logger = logging.getLogger(__name__)

# 1 subscriber あたりのキュー上限。遅いクライアントで `publish` を詰まらせないため、
# 満杯なら古い 1 件を捨てて最新を残す (conversation_store._broadcast_image と同じ戦略)。
_SUBSCRIBER_QUEUE_MAX = 128

# ターン完走後、購読可能なまま保持する秒数。リロードに近いタイミングで再購読したい
# クライアントや、瞬間瞬断後のリトライを救うための短い buffer。
_DEFAULT_RETENTION_SECONDS = 60.0


@dataclass
class ChatTurn:
    """1 つのチャットターンのライフサイクルを表す。

    - `events` は発生した全 SseEvent の replay バッファ。遅刻 subscriber に流す。
    - `subscribers` の Queue に流れる `None` は「これ以上イベントは来ない」の sentinel。
    - `done` は `finish()` 完了後にセットされる。GC タスクはこれを見て retention を待つ。
    """

    turn_id: str
    conversation_id: str
    events: list[SseEvent] = field(default_factory=list)
    subscribers: set[asyncio.Queue[SseEvent | None]] = field(default_factory=set)
    done: asyncio.Event = field(default_factory=asyncio.Event)


class TurnRegistry:
    """`POST /chat` が採番したターンを保持する in-memory レジストリ。

    `POST /chat` はターンを `register()` し、オーケストレータ出力を `publish()` で流す。
    クライアントは別接続 `GET /chat/turns/{turn_id}/events` で `subscribe()` する。
    これにより POST の body 消費と SSE 配信を分離でき、ネットワーク瞬断時に SSE だけ
    再接続する・複数タブで同じターンを観測する、などが自然に実現できる。
    """

    def __init__(self, *, retention_seconds: float = _DEFAULT_RETENTION_SECONDS) -> None:
        self._turns: dict[str, ChatTurn] = {}
        self._retention_seconds = retention_seconds
        self._gc_tasks: set[asyncio.Task[None]] = set()

    def register(self, conversation_id: str) -> ChatTurn:
        """新しいターンを採番してレジストリに登録する。"""
        turn_id = str(uuid.uuid4())
        turn = ChatTurn(turn_id=turn_id, conversation_id=conversation_id)
        self._turns[turn_id] = turn
        return turn

    def get(self, turn_id: str) -> ChatTurn | None:
        return self._turns.get(turn_id)

    def publish(self, turn_id: str, event: SseEvent) -> None:
        """ターンにイベントを記録し、全 subscriber に配信する。

        Queue が満杯なら古い 1 件を捨てて入れ直す (遅い client で全体を止めない)。
        """
        turn = self._turns.get(turn_id)
        if turn is None:
            return
        turn.events.append(event)
        for queue in list(turn.subscribers):
            _put_dropping_oldest(queue, event)

    def finish(self, turn_id: str) -> None:
        """ターン完了。全 subscriber に sentinel を送り、retention 経過後に GC する。"""
        turn = self._turns.get(turn_id)
        if turn is None:
            return
        turn.done.set()
        for queue in list(turn.subscribers):
            _put_dropping_oldest(queue, None)
        task = asyncio.create_task(self._schedule_gc(turn_id))
        self._gc_tasks.add(task)
        task.add_done_callback(self._gc_tasks.discard)

    async def _schedule_gc(self, turn_id: str) -> None:
        try:
            await asyncio.sleep(self._retention_seconds)
        except asyncio.CancelledError:
            return
        self._turns.pop(turn_id, None)

    @asynccontextmanager
    async def subscribe(self, turn_id: str) -> AsyncIterator[AsyncIterator[SseEvent]]:
        """`@asynccontextmanager` で subscribe。yield された iterator を for-await で回す。

        挙動:
        1. まず現時点までの `events` を replay で流す (遅刻 subscriber のため)
        2. 自分の Queue を `subscribers` に登録し、ライブイベントを流す
        3. sentinel `None` を受け取ったら iterator を終了する

        完走済みターンを subscribe した場合でも、replay→sentinel という順序で
        全イベントが届いてから close する。
        """
        turn = self._turns.get(turn_id)
        if turn is None:
            raise KeyError(turn_id)

        queue: asyncio.Queue[SseEvent | None] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
        # 1) replay + 2) 自分を登録 を原子的に行うため、イベントループが他タスクに切り替わる
        # 前に一気に両方やる。ここで publish() が挟まっても、登録後の live publish で拾える
        # ので欠落しない (replay 中に publish が走ると重複の可能性があるが、クライアント側
        # は event に含まれる state で冪等に反映する前提なので問題ない)。
        replayed = list(turn.events)
        turn.subscribers.add(queue)
        already_done = turn.done.is_set()

        async def _iterator() -> AsyncIterator[SseEvent]:
            for ev in replayed:
                yield ev
            if already_done:
                # 完走済みなら sentinel を待つ意味はない (登録前に finish 済みなら
                # Queue は空のまま永遠に待つことになる) ので即終了。
                return
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield item

        try:
            yield _iterator()
        finally:
            turn.subscribers.discard(queue)

    async def shutdown(self) -> None:
        """プロセス終了時に pending な GC タスクを片付ける。"""
        for task in list(self._gc_tasks):
            task.cancel()
        for task in list(self._gc_tasks):
            with suppress(asyncio.CancelledError, Exception):
                await task


def _put_dropping_oldest(queue: asyncio.Queue[SseEvent | None], item: SseEvent | None) -> None:
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        with suppress(asyncio.QueueEmpty):
            queue.get_nowait()
        with suppress(asyncio.QueueFull):
            queue.put_nowait(item)
