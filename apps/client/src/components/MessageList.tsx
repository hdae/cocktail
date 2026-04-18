import { useEffect, useLayoutEffect, useRef } from "react";

import type { Message } from "@cocktail/api-types";

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

// 末尾付近と判定する余裕（ピクセル）。ここを下回るときだけ自動追随する。
const FOLLOW_THRESHOLD_PX = 96;

export function MessageList({ messages, pending, onImageClick }: Props): JSX.Element {
  const endRef = useRef<HTMLDivElement | null>(null);

  const rendered = pending ? [...messages, pending] : messages;
  const isEmpty = rendered.length === 0;

  // ScrollArea の Viewport（実際のスクロールコンテナ）を sentinel から逆引きする。
  // ScrollArea は React 18 の function component で ref を素通ししないため closest を使う。
  const getViewport = (): HTMLElement | null => {
    return (
      endRef.current?.closest<HTMLElement>(
        '[data-slot="scroll-area-viewport"]',
      ) ?? null
    );
  };

  // メッセージ数や pending の切り替わりで即座に末尾へ寄せる。
  // useLayoutEffect を使って DOM 反映直後に実行し、描画前に位置を合わせる。
  useLayoutEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, pending]);

  // 画像読み込み完了で高さが伸びた際にも末尾を追随する。
  // Viewport に capture phase で listener を張り、子の <img> の load を横取り。
  // 「ユーザーが過去を読んでいる途中」は強制しないよう、閾値内にいるときだけ発火。
  useEffect(() => {
    const vp = getViewport();
    if (!vp) return;

    const handleLoad = (ev: Event): void => {
      const target = ev.target;
      if (!(target instanceof HTMLImageElement)) return;
      const nearBottom =
        vp.scrollHeight - vp.scrollTop - vp.clientHeight < FOLLOW_THRESHOLD_PX;
      if (!nearBottom) return;
      endRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
    };

    vp.addEventListener("load", handleLoad, true);
    return () => vp.removeEventListener("load", handleLoad, true);
  }, []);

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
          rendered.map((m) => (
            <MessageRow key={m.id} message={m} onImageClick={onImageClick} />
          ))
        )}
        <div ref={endRef} />
      </div>
    </ScrollArea>
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
