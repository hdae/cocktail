import { useEffect, useRef } from "react";

import type { Message } from "@cocktail/api-types";

import { cn } from "../lib/cn";
import { MessagePart } from "./MessagePart";

interface Props {
  messages: Message[];
  pending: Message | null;
}

export function MessageList({ messages, pending }: Props): JSX.Element {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, pending]);

  const rendered = pending ? [...messages, pending] : messages;

  return (
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto px-4 py-6">
      {rendered.length === 0 ? (
        <div className="m-auto max-w-md text-center text-neutral-500">
          <p className="text-base">日本語で描きたいものを書いてください。</p>
          <p className="mt-1 text-sm opacity-70">
            例: 「猫の女の子、白いワンピース、窓辺、柔らかい光」
          </p>
        </div>
      ) : (
        rendered.map((m) => <MessageRow key={m.id} message={m} />)
      )}
      <div ref={endRef} />
    </div>
  );
}

function MessageRow({ message }: { message: Message }): JSX.Element {
  const isUser = message.role === "user";
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-3",
          isUser ? "bg-blue-600 text-white" : "bg-neutral-900 text-neutral-100",
        )}
      >
        {message.parts.map((part, i) => (
          <MessagePart key={i} part={part} />
        ))}
      </div>
    </div>
  );
}
