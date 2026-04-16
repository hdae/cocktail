import { create } from "zustand";

import type {
  ChatRequest,
  ContentPart,
  ImagePart,
  Message,
  SseEvent,
  UserContentPart,
} from "@cocktail/api-types";

import { streamChat } from "../lib/sse";

type Status = "idle" | "streaming" | "error";

interface PendingToolCall {
  callId: string;
  name: string;
  args: Record<string, unknown>;
  status: "running" | "done" | "error";
  summary?: string;
}

interface PendingAssistant {
  messageId: string;
  reasoning: string;
  toolCalls: Record<string, PendingToolCall>;
  images: ImagePart[];
}

interface ChatState {
  conversationId: string | null;
  messages: Message[];
  pending: PendingAssistant | null;
  status: Status;
  error: string | null;
  abort: AbortController | null;

  sendMessage: (parts: UserContentPart[]) => Promise<void>;
  reset: () => void;
}

export const useChatStore = create<ChatState>()((set, get) => ({
  conversationId: null,
  messages: [],
  pending: null,
  status: "idle",
  error: null,
  abort: null,

  reset: () => {
    get().abort?.abort();
    set({
      conversationId: null,
      messages: [],
      pending: null,
      status: "idle",
      error: null,
      abort: null,
    });
  },

  sendMessage: async (parts) => {
    const prev = get().abort;
    if (prev) prev.abort();

    const abort = new AbortController();
    const conversationId = get().conversationId;

    const req: ChatRequest = {
      conversation_id: conversationId,
      parts,
      parent_id: null,
    };

    set({ status: "streaming", error: null, abort, pending: null });

    try {
      for await (const ev of streamChat(req, abort.signal)) {
        applyEvent(ev);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      set({ status: "error", error: msg, abort: null, pending: null });
      return;
    }

    if (get().status === "streaming") {
      set({ status: "idle", abort: null });
    }
  },
}));

function applyEvent(ev: SseEvent): void {
  const setState = useChatStore.setState;
  switch (ev.type) {
    case "conversation":
      setState({ conversationId: ev.conversation_id });
      return;
    case "user_saved":
      setState((s) => ({ messages: [...s.messages, ev.message] }));
      return;
    case "assistant_start":
      setState({
        pending: {
          messageId: ev.message_id,
          reasoning: "",
          toolCalls: {},
          images: [],
        },
      });
      return;
    case "text_delta":
      setState((s) => {
        if (!s.pending) return {};
        return {
          pending: { ...s.pending, reasoning: s.pending.reasoning + ev.delta },
        };
      });
      return;
    case "tool_call_start":
      setState((s) => {
        if (!s.pending) return {};
        return {
          pending: {
            ...s.pending,
            toolCalls: {
              ...s.pending.toolCalls,
              [ev.call_id]: {
                callId: ev.call_id,
                name: ev.name,
                args: ev.args,
                status: "running",
              },
            },
          },
        };
      });
      return;
    case "image_ready":
      setState((s) => {
        if (!s.pending) return {};
        const image: ImagePart = {
          type: "image",
          image_id: ev.image_id,
          mime: ev.mime,
          width: ev.width,
          height: ev.height,
        };
        return { pending: { ...s.pending, images: [...s.pending.images, image] } };
      });
      return;
    case "tool_call_end":
      setState((s) => {
        if (!s.pending) return {};
        const existing = s.pending.toolCalls[ev.call_id];
        if (!existing) return {};
        return {
          pending: {
            ...s.pending,
            toolCalls: {
              ...s.pending.toolCalls,
              [ev.call_id]: {
                ...existing,
                status: ev.status,
                summary: ev.summary,
              },
            },
          },
        };
      });
      return;
    case "assistant_end":
      setState((s) => ({
        messages: [...s.messages, ev.message],
        pending: null,
      }));
      return;
    case "error":
      setState({ status: "error", error: `${ev.code}: ${ev.message}` });
      return;
    case "done":
      if (useChatStore.getState().status === "streaming") {
        setState({ status: "idle", abort: null });
      }
      return;
  }
}

/** `pending` を仮想 `Message` に変換して `MessageList` で扱いやすくする。 */
export function pendingAsMessage(
  conversationId: string | null,
  pending: PendingAssistant | null,
): Message | null {
  if (!conversationId || !pending) return null;
  const parts: ContentPart[] = [];
  if (pending.reasoning) {
    parts.push({ type: "text", text: pending.reasoning });
  }
  for (const tc of Object.values(pending.toolCalls)) {
    parts.push({
      type: "tool_call",
      id: tc.callId,
      name: tc.name,
      args: tc.args,
      status: tc.status === "running" ? "running" : tc.status,
    });
  }
  for (const img of pending.images) {
    parts.push(img);
  }
  if (parts.length === 0) {
    // reasoning も tool_call もまだ来ていない瞬間の placeholder
    parts.push({ type: "text", text: "…" });
  }
  return {
    id: pending.messageId,
    conversation_id: conversationId,
    role: "assistant",
    parts,
    created_at: new Date().toISOString(),
    parent_id: null,
  };
}
