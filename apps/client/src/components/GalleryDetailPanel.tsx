import { useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";

import type { GeneratedImageRef } from "@cocktail/api-types";

import { useChatStore } from "../store/chat";

interface Props {
  images: GeneratedImageRef[];
  index: number;
  onIndexChange: (index: number) => void;
  onClose: () => void;
}

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
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-6"
      onClick={onClose}
    >
      <div
        className="relative flex max-h-full w-full max-w-4xl flex-col gap-4 overflow-y-auto rounded-xl bg-neutral-900 p-5 text-neutral-100"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={onClose}
          className="absolute right-3 top-3 rounded-md px-2 py-1 text-xs text-neutral-400 transition hover:bg-neutral-800 hover:text-neutral-200"
          aria-label="閉じる"
        >
          閉じる
        </button>

        <div className="relative flex justify-center">
          <img
            src={image.image_url}
            alt={image.prompt}
            className="max-h-[70vh] max-w-full rounded-lg object-contain"
          />
          <button
            type="button"
            onClick={() => onIndexChange(index - 1)}
            disabled={!hasPrev}
            aria-label="前の画像"
            className="absolute left-2 top-1/2 -translate-y-1/2 rounded-full bg-black/50 px-3 py-2 text-lg text-neutral-100 transition hover:bg-black/70 disabled:cursor-not-allowed disabled:opacity-20"
          >
            ‹
          </button>
          <button
            type="button"
            onClick={() => onIndexChange(index + 1)}
            disabled={!hasNext}
            aria-label="次の画像"
            className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-black/50 px-3 py-2 text-lg text-neutral-100 transition hover:bg-black/70 disabled:cursor-not-allowed disabled:opacity-20"
          >
            ›
          </button>
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

        <div className="flex justify-end gap-2">
          <a
            href={image.image_url}
            download={`${image.image_id}.webp`}
            className="rounded-md bg-neutral-800 px-3 py-1.5 text-xs font-medium text-neutral-100 transition hover:bg-neutral-700"
          >
            ダウンロード
          </a>
          <button
            type="button"
            onClick={usePrompt}
            className="rounded-md bg-neutral-100 px-3 py-1.5 text-xs font-medium text-neutral-900 transition hover:bg-white"
          >
            このプロンプトを使う
          </button>
        </div>
      </div>
    </div>
  );
}
