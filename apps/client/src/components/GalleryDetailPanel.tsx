import { useNavigate } from "@tanstack/react-router"
import { ChevronLeft, ChevronRight, Download, Wand2, X } from "lucide-react"
import { useEffect } from "react"

import type { GeneratedImageRef } from "@cocktail/api-types"

import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

import { useChatStore } from "../store/chat"

interface Props {
  images: GeneratedImageRef[];
  index: number;
  onIndexChange: (index: number) => void;
  onClose: () => void;
}

// ヘッダー内の各アイコンボタン共通クラス。TooltipTrigger を直接 button 化して、
// render prop を二重に挟まないことで cloneElement 経由の props 合流を安定させる。
const iconBtn = cn(
  "inline-flex size-7 items-center justify-center rounded-md text-neutral-300",
  "outline-none transition hover:bg-neutral-800 hover:text-neutral-100",
  "focus-visible:ring-2 focus-visible:ring-neutral-500",
  "disabled:pointer-events-none disabled:opacity-40",
  "aria-disabled:cursor-not-allowed aria-disabled:opacity-40",
  "[&_svg]:size-4",
);

export function GalleryDetailPanel({
  images,
  index,
  onIndexChange,
  onClose,
}: Props): JSX.Element | null {
  const navigate = useNavigate();
  const setDraft = useChatStore((s) => s.setDraft);

  const image = images[index];
  const hasPrev = index > 0;
  const hasNext = index < images.length - 1;

  // ←/→ で前後移動、Esc で閉じる。モーダル内に入力欄は無いので無条件に
  // document で受ける。開いてる間だけ listener を張る。
  useEffect(() => {
    const handler = (e: KeyboardEvent): void => {
      if (e.key === "ArrowLeft" && hasPrev) {
        e.preventDefault();
        onIndexChange(index - 1);
      } else if (e.key === "ArrowRight" && hasNext) {
        e.preventDefault();
        onIndexChange(index + 1);
      } else if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [index, hasPrev, hasNext, onIndexChange, onClose]);

  if (!image) return null;

  const usePrompt = (): void => {
    setDraft("new", image.prompt);
    void navigate({ to: "/conversations/new" });
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6"
      onClick={onClose}
    >
      {/* 高さは max-h ではなく h で確定値を与える。max-h だと flex コンテナの
          高さが content-based になり、flex-1 な ScrollArea が 0 に潰れて
          Viewport の overflow: scroll が効かなくなる。 */}
      <div
        className="flex h-[calc(100svh-3rem)] w-full max-w-4xl flex-col overflow-hidden rounded-xl bg-neutral-900 text-neutral-100"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex shrink-0 items-center gap-1 border-b border-neutral-800 px-3 py-2">
          <Tooltip>
            <TooltipTrigger
              className={iconBtn}
              onClick={usePrompt}
              aria-label="このプロンプトを使う"
            >
              <Wand2 />
            </TooltipTrigger>
            <TooltipContent>このプロンプトを使う</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger
              className={iconBtn}
              aria-label="ダウンロード"
              render={
                <a
                  href={image.image_url}
                  download={`${image.image_id}.webp`}
                />
              }
            >
              <Download />
            </TooltipTrigger>
            <TooltipContent>ダウンロード</TooltipContent>
          </Tooltip>

          <div className="flex-1" />

          <Tooltip>
            <TooltipTrigger
              className={iconBtn}
              aria-disabled={!hasPrev}
              onClick={() => {
                if (hasPrev) onIndexChange(index - 1);
              }}
              aria-label="前の画像"
            >
              <ChevronLeft />
            </TooltipTrigger>
            <TooltipContent>前の画像</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger
              className={iconBtn}
              aria-disabled={!hasNext}
              onClick={() => {
                if (hasNext) onIndexChange(index + 1);
              }}
              aria-label="次の画像"
            >
              <ChevronRight />
            </TooltipTrigger>
            <TooltipContent>次の画像</TooltipContent>
          </Tooltip>

          <div className="flex-1" />

          <Tooltip>
            <TooltipTrigger
              className={iconBtn}
              onClick={onClose}
              aria-label="閉じる"
            >
              <X />
            </TooltipTrigger>
            <TooltipContent>閉じる</TooltipContent>
          </Tooltip>
        </header>

        <ScrollArea className="min-h-0 flex-1">
          <div className="flex flex-col gap-4 p-5">
            <div className="flex justify-center">
              <img
                src={image.image_url}
                alt={image.prompt}
                className="max-h-[60vh] max-w-full rounded-lg object-contain"
              />
            </div>

            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-xs text-neutral-300">
              <dt className="text-neutral-500">prompt</dt>
              <dd className="whitespace-pre-wrap break-words font-mono text-[11px] leading-5 text-neutral-200">
                {image.prompt}
              </dd>
              <dt className="text-neutral-500">seed</dt>
              <dd className="font-mono">{image.seed}</dd>
              <dt className="text-neutral-500">aspect</dt>
              <dd>
                {image.aspect_ratio} ({image.width}×{image.height})
              </dd>
              <dt className="text-neutral-500">cfg</dt>
              <dd>{image.cfg_preset}</dd>
              <dt className="text-neutral-500">created</dt>
              <dd className="font-mono text-[11px]">{image.created_at}</dd>
              <dt className="text-neutral-500">conversation</dt>
              <dd className="font-mono text-[11px]">
                {image.conversation_id.slice(0, 8)}
              </dd>
            </dl>
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}
