import type { GeneratedImageRef } from "@cocktail/api-types";

import { useChatStore } from "../store/chat";

interface Props {
  image: GeneratedImageRef;
  onClose: () => void;
}

export function GalleryDetailPanel({ image, onClose }: Props): JSX.Element {
  const setView = useChatStore((s) => s.setView);
  const setComposerDraft = useChatStore((s) => s.setComposerDraft);

  const sendToComposer = (): void => {
    setComposerDraft(image.prompt_excerpt);
    setView("chat");
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

        <div className="flex justify-center">
          <img
            src={image.image_url}
            alt={image.prompt_excerpt}
            className="max-h-[70vh] max-w-full rounded-lg object-contain"
          />
        </div>

        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-xs text-neutral-300">
          <dt className="text-neutral-500">prompt</dt>
          <dd className="break-words font-mono text-[11px] leading-5 text-neutral-200">
            {image.prompt_excerpt}
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
          <button
            type="button"
            onClick={sendToComposer}
            className="rounded-md bg-neutral-100 px-3 py-1.5 text-xs font-medium text-neutral-900 transition hover:bg-white"
          >
            この prompt をコンポーザに流す
          </button>
        </div>
      </div>
    </div>
  );
}
