import { fetchEventSource } from "@microsoft/fetch-event-source";

import {
  type GeneratedImageRef,
  GeneratedImageRefSchema,
  type SseEvent,
  SseEventSchema,
} from "@cocktail/api-types";

/**
 * `GET /chat/turns/{turn_id}/events` を購読し、SSE イベントを順次受信する。
 *
 * ターンは `POST /chat` で先に採番されている前提。POST と SSE を別接続にすることで、
 * navigate / hydrate で接続が切れても replay で欠落なく再購読できる（retention 内）。
 *
 * 本家 `EventSource` は 404 を自動再接続してしまうので `fetchEventSource` を使い、
 * `onopen` で非 2xx を即 throw する。
 */
export async function* subscribeTurn(
  turnId: string,
  signal: AbortSignal,
): AsyncGenerator<SseEvent> {
  const queue: SseEvent[] = [];
  let resolveNext: (() => void) | null = null;
  let done = false;
  let error: unknown = null;

  const push = (ev: SseEvent) => {
    queue.push(ev);
    if (resolveNext) {
      const r = resolveNext;
      resolveNext = null;
      r();
    }
  };

  const finish = (err?: unknown) => {
    done = true;
    if (err !== undefined) error = err;
    if (resolveNext) {
      const r = resolveNext;
      resolveNext = null;
      r();
    }
  };

  const url = `/api/chat/turns/${encodeURIComponent(turnId)}/events`;
  const promise = fetchEventSource(url, {
    method: "GET",
    signal,
    openWhenHidden: true,
    onopen: async (res) => {
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      }
    },
    onmessage: (msg) => {
      if (!msg.data) return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(msg.data);
      } catch {
        return;
      }
      const result = SseEventSchema.safeParse(parsed);
      if (!result.success) {
        console.warn("Dropping unparseable SSE event", parsed, result.error);
        return;
      }
      push(result.data);
    },
    onerror: (err) => {
      // 返さないと fetch-event-source は自動再試行する。ここで throw すると上に伝わる。
      throw err;
    },
    onclose: () => {
      finish();
    },
  })
    .catch((err: unknown) => {
      finish(err);
    });

  try {
    while (true) {
      const next = queue.shift();
      if (next !== undefined) {
        yield next;
        continue;
      }
      if (done) break;
      await new Promise<void>((resolve) => {
        resolveNext = resolve;
      });
    }
    if (error) throw error;
  } finally {
    await promise;
  }
}

export interface ImageEventsOptions {
  /** 新しい `GeneratedImageRef` が broadcast された時に呼ばれる。 */
  onCreate: (ref: GeneratedImageRef) => void;
  /**
   * (再)接続成功時に呼ばれる。呼出側は初回/再接続を自前でカウントし、
   * 再接続時には一覧 API を再取得して欠落を埋めるのに使う。
   */
  onOpen?: () => void;
  signal: AbortSignal;
}

/**
 * GET /images/events を購読し、broadcast された `GeneratedImageRef` を受け取る。
 *
 * サーバは接続維持のため 15 秒おきに `: ping` コメントを送ってくる。
 * fetchEventSource は transport エラー時に自動再接続するので、呼び出し側は
 * signal で明示的に abort するまで中断を気にしなくて良い。
 */
export function streamImageEvents(options: ImageEventsOptions): Promise<void> {
  const { onCreate, onOpen, signal } = options;
  return fetchEventSource("/api/images/events", {
    method: "GET",
    signal,
    openWhenHidden: true,
    onopen: async (res) => {
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      }
      onOpen?.();
    },
    onmessage: (msg) => {
      if (msg.event !== "image_created" || !msg.data) return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(msg.data);
      } catch {
        return;
      }
      const result = GeneratedImageRefSchema.safeParse(parsed);
      if (!result.success) {
        console.warn("Dropping unparseable image event", parsed, result.error);
        return;
      }
      onCreate(result.data);
    },
    onerror: (err) => {
      // throw せず return undefined → ライブラリが自動再接続する。
      console.warn("image events stream error; reconnecting", err);
    },
  });
}
