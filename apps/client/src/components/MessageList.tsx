import { useEffect, useRef } from "react";

import type { Message } from "@cocktail/api-types";

import { cn } from "../lib/utils";
import { MessagePart } from "./MessagePart";
import { ScrollArea } from "./ui/scroll-area";

interface Props {
  messages: Message[];
  pending: Message | null;
}

const EXAMPLES = [
  "夕暮れの教室で、窓際の席に座って文庫本を読んでいる女の子を生成してください。外からのやわらかな光が頬と前髪を照らしている感じでお願いします。",
  "星空の下で、猫耳の少女が静かに微笑んでいる縦長の絵にしてください。髪はピンクで、背景には流れ星を一つだけ入れてほしいです。",
  "雨上がりの街角で、傘を閉じながら歩く女の子。濡れた石畳に街灯の光が反射していて、少し物思いに耽った表情が良いです。",
];

export function MessageList({ messages, pending }: Props): JSX.Element {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, pending]);

  const rendered = pending ? [...messages, pending] : messages;
  const isEmpty = rendered.length === 0;

  return (
    <ScrollArea className="min-h-0 flex-1 overflow-hidden">
      <div
        className={cn(
          "mx-auto flex max-w-5xl flex-col gap-5 px-5 py-8",
          isEmpty && "h-full justify-center",
        )}
      >
        {isEmpty ? (
          <div className="mx-auto w-full max-w-xl text-neutral-400">
            <p className="text-base text-neutral-200">
              どんな絵を生成しましょうか。
            </p>
            <p className="mt-1 text-sm text-neutral-500">
              思い浮かべているシーンを日本語の文章でそのまま伝えてください。
              アスペクト比やタッチの希望があれば一緒に書いてもらえると反映します。
            </p>
            <ul className="mt-5 space-y-2">
              {EXAMPLES.map((ex) => (
                <li
                  key={ex}
                  className="rounded-lg bg-neutral-900/60 px-4 py-3 text-sm leading-relaxed text-neutral-400"
                >
                  {ex}
                </li>
              ))}
            </ul>
          </div>
        ) : (
          rendered.map((m) => <MessageRow key={m.id} message={m} />)
        )}
        <div ref={endRef} />
      </div>
    </ScrollArea>
  );
}

function MessageRow({ message }: { message: Message }): JSX.Element {
  const isUser = message.role === "user";
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] space-y-2 rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-neutral-100 text-neutral-900"
            : "bg-neutral-900/80 text-neutral-100",
        )}
      >
        {message.parts.map((part, i) => (
          <MessagePart key={i} part={part} />
        ))}
      </div>
    </div>
  );
}
