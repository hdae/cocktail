import { useCallback, useEffect, useState, type KeyboardEvent } from "react";

import type { UserContentPart } from "@cocktail/api-types";

import { useChatStore } from "../store/chat";

interface Props {
  disabled?: boolean;
  onSend: (parts: UserContentPart[]) => void;
}

const PLACEHOLDER =
  "生成したい絵の様子を日本語でそのまま書いてください。Cmd/Ctrl + Enter で送信します。";

export function ComposerInput({ disabled, onSend }: Props): JSX.Element {
  const [text, setText] = useState("");
  const composerDraft = useChatStore((s) => s.composerDraft);
  const setComposerDraft = useChatStore((s) => s.setComposerDraft);

  useEffect(() => {
    if (composerDraft) {
      setText(composerDraft);
      setComposerDraft("");
    }
  }, [composerDraft, setComposerDraft]);

  const send = useCallback(() => {
    const body = text.trim();
    if (!body) return;
    onSend([{ type: "text", text: body }]);
    setText("");
  }, [text, onSend]);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        send();
      }
    },
    [send],
  );

  const canSend = !disabled && text.trim().length > 0;

  return (
    <div className="flex items-end gap-2">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={PLACEHOLDER}
        rows={1}
        className="min-h-10 max-h-40 flex-1 resize-none rounded-lg bg-neutral-900 px-3.5 py-2 text-sm leading-6 text-neutral-100 placeholder:text-neutral-500 focus:outline-none focus:ring-1 focus:ring-neutral-600 disabled:opacity-50"
        style={{ fieldSizing: "content" } as React.CSSProperties}
        disabled={disabled}
      />
      <button
        type="button"
        onClick={send}
        disabled={!canSend}
        className="h-10 shrink-0 rounded-lg bg-neutral-100 px-4 text-sm font-medium text-neutral-900 transition hover:bg-white disabled:cursor-not-allowed disabled:bg-neutral-800 disabled:text-neutral-500"
      >
        送信
      </button>
    </div>
  );
}
