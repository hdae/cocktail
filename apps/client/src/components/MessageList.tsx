import { ChevronDown } from "lucide-react";
import { useLayoutEffect, useRef } from "react";

import type { Message } from "@cocktail/api-types";

import { useAutoScrollToBottom } from "../hooks/use-auto-scroll-to-bottom";
import { cn } from "../lib/utils";
import { MessagePart } from "./MessagePart";
import { ScrollArea } from "./ui/scroll-area";

interface Props {
  messages: Message[];
  pending: Message | null;
  onImageClick?: ((imageId: string) => void) | undefined;
}

const EXAMPLES = [
  "夕暮れの教室で、窓際の席に座って文庫本を読んでいる女の子を生成してください。外からのやわらかな光が頬と前髪を照らしている感じでお願いします。",
  "星空の下で、猫耳の少女が静かに微笑んでいる縦長の絵にしてください。髪はピンクで、背景には流れ星を一つだけ入れてほしいです。",
  "雨上がりの街角で、傘を閉じながら歩く女の子。濡れた石畳に街灯の光が反射していて、少し物思いに耽った表情が良いです。",
];

export function MessageList({ messages, pending, onImageClick }: Props): JSX.Element {
  const { anchorRef, isAtBottom, scrollToBottom, followIfNeeded } =
    useAutoScrollToBottom();
  const prevMessagesLenRef = useRef(messages.length);

  const rendered = pending ? [...messages, pending] : messages;
  const isEmpty = rendered.length === 0;

  // 末尾追随の発火ポリシー:
  // - メッセージ件数が増えた瞬間（user 送信 / assistant 完了）は smooth
  // - それ以外（pending の text_delta / image_ready）は instant
  // streaming は高頻度で scrollHeight が伸びるため smooth を重ねると目標が動き続けて
  // 破綻する。boundary のみ smooth にして気持ちよさを確保する。
  useLayoutEffect(() => {
    const lenChanged = messages.length !== prevMessagesLenRef.current;
    prevMessagesLenRef.current = messages.length;
    followIfNeeded(lenChanged);
  }, [messages.length, pending, followIfNeeded]);

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <ScrollArea className="flex-1 overflow-hidden">
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
            rendered.map((m) => (
              <MessageRow key={m.id} message={m} onImageClick={onImageClick} />
            ))
          )}
          <div ref={anchorRef} />
        </div>
      </ScrollArea>
      {!isAtBottom && (
        <button
          type="button"
          onClick={() => scrollToBottom(true)}
          aria-label="最新のメッセージへ移動"
          className="absolute bottom-4 left-1/2 z-10 flex h-9 w-9 -translate-x-1/2 items-center justify-center rounded-full bg-neutral-800/95 text-neutral-200 shadow-lg ring-1 ring-neutral-700 backdrop-blur transition hover:bg-neutral-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-400"
        >
          <ChevronDown className="h-5 w-5" aria-hidden="true" />
        </button>
      )}
    </div>
  );
}

function MessageRow({
  message,
  onImageClick,
}: {
  message: Message;
  onImageClick?: ((imageId: string) => void) | undefined;
}): JSX.Element {
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
          <MessagePart key={i} part={part} onImageClick={onImageClick} />
        ))}
      </div>
    </div>
  );
}
