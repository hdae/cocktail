import { useNavigate } from "@tanstack/react-router"
import { ChevronLeft, ChevronRight, Download, Info, Wand2, X } from "lucide-react"
import { useEffect, useState } from "react"

import type { GeneratedImageRef } from "@cocktail/api-types"

import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

import { useChatStore } from "../../store/chat"

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

// 画像左右に重ねる前後ナビ。大きめヒットエリア + 半透明背景で視認性を確保。
const sideNavBtn = cn(
  "absolute top-1/2 -translate-y-1/2 z-10",
  "inline-flex size-10 items-center justify-center rounded-full",
  "bg-neutral-900/70 text-neutral-200 backdrop-blur",
  "outline-none transition hover:bg-neutral-800 hover:text-neutral-50",
  "focus-visible:ring-2 focus-visible:ring-neutral-500",
  "aria-disabled:pointer-events-none aria-disabled:opacity-30",
  "[&_svg]:size-5",
);

export function ImageViewer({
  images,
  index,
  onIndexChange,
  onClose,
}: Props): JSX.Element | null {
  const navigate = useNavigate();
  const setDraft = useChatStore((s) => s.setDraft);
  const [detailsOpen, setDetailsOpen] = useState(false);

  const image = images[index];
  const hasPrev = index > 0;
  const hasNext = index < images.length - 1;

  // ←/→ で前後移動。Esc は Drawer が開いていれば Drawer を先に閉じる。
  useEffect(() => {
    const handler = (e: KeyboardEvent): void => {
      if (e.key === "ArrowLeft" && hasPrev) {
        e.preventDefault();
        onIndexChange(index - 1);
      } else if (e.key === "ArrowRight" && hasNext) {
        e.preventDefault();
        onIndexChange(index + 1);
      } else if (e.key === "Escape") {
        if (detailsOpen) {
          setDetailsOpen(false);
        } else {
          onClose();
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [index, hasPrev, hasNext, onIndexChange, onClose, detailsOpen]);

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
        className="relative flex h-[calc(100svh-3rem)] w-full max-w-4xl flex-col overflow-hidden rounded-xl bg-neutral-900 text-neutral-100"
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

          <Tooltip>
            <TooltipTrigger
              className={cn(iconBtn, detailsOpen && "bg-neutral-800 text-neutral-100")}
              onClick={() => setDetailsOpen((v) => !v)}
              aria-label="詳細を表示"
              aria-expanded={detailsOpen}
            >
              <Info />
            </TooltipTrigger>
            <TooltipContent>詳細</TooltipContent>
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

        {/* メイン領域はダイアログの残り縦幅をフルに使う。min-h-0 を忘れると
            flex コンテナが content-based 高さになり max-h-full が効かない。
            前後ナビは画像を内包するこの領域を relative 基準に absolute 配置。 */}
        <div className="relative flex min-h-0 flex-1 items-center justify-center p-5">
          <img
            src={image.image_url}
            alt={image.prompt}
            className="max-h-full max-w-full rounded-lg object-contain"
          />

          <button
            type="button"
            className={cn(sideNavBtn, "left-2")}
            disabled={!hasPrev}
            aria-disabled={!hasPrev}
            aria-label="前の画像"
            onClick={() => {
              if (hasPrev) onIndexChange(index - 1);
            }}
          >
            <ChevronLeft />
          </button>
          <button
            type="button"
            className={cn(sideNavBtn, "right-2")}
            disabled={!hasNext}
            aria-disabled={!hasNext}
            aria-label="次の画像"
            onClick={() => {
              if (hasNext) onIndexChange(index + 1);
            }}
          >
            <ChevronRight />
          </button>
        </div>

        {/* ダイアログ内ボトムドロワー。max-h を超えない範囲で下からスライドアップ。
            閉時は translate-y-full + pointer-events-none で完全に退避する。 */}
        <div
          role="region"
          aria-label="画像の詳細"
          aria-hidden={!detailsOpen}
          className={cn(
            "absolute inset-x-0 bottom-0 z-10 flex max-h-[50%] flex-col",
            "border-t border-neutral-800 bg-neutral-900/95 backdrop-blur",
            "transition-transform duration-200 ease-out",
            detailsOpen ? "translate-y-0" : "pointer-events-none translate-y-full",
          )}
        >
          <div className="flex shrink-0 items-center justify-between border-b border-neutral-800 px-4 py-2">
            <span className="text-xs font-medium tracking-wide text-neutral-300">
              詳細
            </span>
            <button
              type="button"
              className={iconBtn}
              onClick={() => setDetailsOpen(false)}
              aria-label="詳細を閉じる"
            >
              <X />
            </button>
          </div>
          <ScrollArea className="min-h-0 flex-1">
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 px-4 py-3 text-xs text-neutral-300">
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
          </ScrollArea>
        </div>
      </div>
    </div>
  );
}
