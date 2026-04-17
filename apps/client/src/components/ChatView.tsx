import { useMemo } from "react";

import { pendingAsMessage, useChatStore } from "../store/chat";
import { ComposerInput } from "./ComposerInput";
import { HealthBanner } from "./HealthBanner";
import { MessageList } from "./MessageList";

export function ChatView(): JSX.Element {
  const {
    conversationId,
    messages,
    pending,
    status,
    error,
    sendMessage,
    reset,
    setView,
  } = useChatStore();

  const pendingMessage = useMemo(
    () => pendingAsMessage(conversationId, pending),
    [conversationId, pending],
  );

  return (
    <div className="flex h-full flex-col bg-neutral-950 text-neutral-100">
      <header className="flex items-center justify-between px-5 py-3">
        <h1 className="text-sm font-semibold tracking-wide text-neutral-200">
          Cocktail
        </h1>
        <div className="flex items-center gap-3 text-xs text-neutral-500">
          {conversationId && (
            <span className="font-mono">{conversationId.slice(0, 8)}</span>
          )}
          <button
            type="button"
            onClick={() => setView("gallery")}
            className="rounded-md px-2.5 py-1 text-xs text-neutral-400 transition hover:bg-neutral-900 hover:text-neutral-200"
          >
            ギャラリー
          </button>
          <button
            type="button"
            onClick={reset}
            className="rounded-md px-2.5 py-1 text-xs text-neutral-400 transition hover:bg-neutral-900 hover:text-neutral-200"
          >
            新しい会話
          </button>
        </div>
      </header>
      <MessageList messages={messages} pending={pendingMessage} />
      <div className="bg-neutral-950 px-5 pb-5 pt-2">
        <div className="mx-auto flex max-w-5xl flex-col gap-2">
          <HealthBanner />
          {error && (
            <div className="rounded-md bg-red-950/40 px-3 py-2 text-xs text-red-200">
              {error}
            </div>
          )}
          <ComposerInput
            disabled={status === "streaming"}
            onSend={(parts) => {
              void sendMessage(parts);
            }}
          />
        </div>
      </div>
    </div>
  );
}
