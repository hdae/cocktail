import { useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo } from "react";

import { SidebarTrigger } from "@/components/ui/sidebar";

import { pendingAsMessage, useChatStore } from "../store/chat";
import { ComposerInput } from "./ComposerInput";
import { HealthBanner } from "./HealthBanner";
import { MessageList } from "./MessageList";

interface Props {
  /** `"new"` = 未開始チャット / UUID = 既存会話（loader が store を hydrate 済み） */
  conversationId: string;
}

export function ChatView({ conversationId }: Props): JSX.Element {
  const navigate = useNavigate();
  const storeConversationId = useChatStore((s) => s.conversationId);
  const messages = useChatStore((s) => s.messages);
  const pending = useChatStore((s) => s.pending);
  const status = useChatStore((s) => s.status);
  const error = useChatStore((s) => s.error);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const reset = useChatStore((s) => s.reset);
  const promoteDraft = useChatStore((s) => s.promoteDraft);

  // `/conversations/new` に入ったら古い会話 state を破棄する。
  useEffect(() => {
    if (conversationId === "new" && useChatStore.getState().conversationId !== null) {
      reset();
    }
  }, [conversationId, reset]);

  // 未開始チャットで送信 → サーバが UUID を返した瞬間に URL を差し替える。
  // useEffect + selector だと、/new 画面に入った直後の reset() 反映前に
  // 古い storeConversationId が閉じ込められて誤 navigate してしまう。
  // そのため null → 非null の遷移だけに反応する subscribe を使う。
  useEffect(() => {
    if (conversationId !== "new") return;
    const unsubscribe = useChatStore.subscribe((state, prevState) => {
      if (prevState.conversationId !== null) return;
      if (state.conversationId === null) return;
      promoteDraft(state.conversationId);
      void navigate({
        to: "/conversations/$conversationId",
        params: { conversationId: state.conversationId },
        replace: true,
      });
    });
    return unsubscribe;
  }, [conversationId, navigate, promoteDraft]);

  // 既存会話タブ間で直接遷移した際に store が古い場合、loader が新しい id で hydrate 済み
  // なので storeConversationId 側を信頼する。表示上の id はどちらも同じ UUID になるはず。
  const activeConversationId =
    conversationId === "new" ? storeConversationId : conversationId;

  const pendingMessage = useMemo(
    () => pendingAsMessage(activeConversationId, pending),
    [activeConversationId, pending],
  );

  return (
    <div className="flex h-full flex-col bg-neutral-950 text-neutral-100">
      <header className="flex items-center justify-between gap-2 px-5 py-3">
        <div className="flex items-center gap-2">
          <SidebarTrigger className="text-neutral-400" />
          <h1 className="text-sm font-semibold tracking-wide text-neutral-200">
            Cocktail
          </h1>
        </div>
        <div className="flex items-center gap-3 text-xs text-neutral-500">
          {activeConversationId && (
            <span className="font-mono">{activeConversationId.slice(0, 8)}</span>
          )}
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
            conversationId={conversationId}
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
