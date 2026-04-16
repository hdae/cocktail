import { fetchEventSource } from "@microsoft/fetch-event-source";

import { type ChatRequest, type SseEvent, SseEventSchema } from "@cocktail/api-types";

/**
 * POST /chat を叩いて SSE を受信する。呼び出し側は `for await` で順次受け取れる。
 *
 * 本家 `EventSource` は POST も JSON ボディも許さないので、`@microsoft/fetch-event-source`
 * を使う。SSE フレームの `event:` が Zod 判別 union の discriminator と一致しているため
 * `data` の JSON だけでパースしても型が決まる。
 */
export async function* streamChat(
  req: ChatRequest,
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

  const promise = fetchEventSource("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
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
      if (queue.length > 0) {
        yield queue.shift()!;
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
