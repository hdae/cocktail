import { useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useState } from "react";

import type { ConversationSummary } from "@cocktail/api-types";

import { SidebarTrigger } from "@/components/ui/sidebar";

import { listConversations } from "../lib/api";

type Status = "idle" | "loading" | "error";

export function HistoryView(): JSX.Element {
  const navigate = useNavigate();
  const [items, setItems] = useState<ConversationSummary[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setStatus("loading");
    setError(null);
    try {
      const rows = await listConversations();
      setItems(rows);
      setStatus("idle");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="flex h-full flex-col bg-neutral-950 text-neutral-100">
      <header className="flex items-center gap-2 px-5 py-3">
        <SidebarTrigger className="text-neutral-400" />
        <h1 className="text-sm font-semibold tracking-wide text-neutral-200">
          チャット履歴
        </h1>
      </header>

      <div className="flex-1 overflow-y-auto px-5 pb-5">
        {error && (
          <div className="mb-3 rounded-md bg-red-950/40 px-3 py-2 text-xs text-red-200">
            {error}
          </div>
        )}
        {items.length === 0 && status !== "loading" && !error && (
          <div className="mt-10 text-center text-xs text-neutral-500">
            まだ会話はありません。
          </div>
        )}
        <ul className="mx-auto flex max-w-3xl flex-col divide-y divide-neutral-900">
          {items.map((conv) => (
            <li key={conv.id}>
              <button
                type="button"
                onClick={() =>
                  void navigate({
                    to: "/conversations/$conversationId",
                    params: { conversationId: conv.id },
                  })
                }
                className="flex w-full flex-col gap-1 px-2 py-3 text-left transition hover:bg-neutral-900"
              >
                <div className="truncate text-sm text-neutral-100">
                  {conv.title}
                </div>
                <div className="flex items-center gap-3 text-[11px] text-neutral-500">
                  <span className="font-mono">{conv.id.slice(0, 8)}</span>
                  <span>{conv.message_count} 件</span>
                  <span className="font-mono">{conv.updated_at}</span>
                </div>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
