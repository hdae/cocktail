import { useCallback, useEffect, useState } from "react";

import type { GeneratedImageRef } from "@cocktail/api-types";

import { listGeneratedImages } from "../lib/api";
import { useChatStore } from "../store/chat";
import { GalleryDetailPanel } from "./GalleryDetailPanel";

export function GalleryView(): JSX.Element {
  const setView = useChatStore((s) => s.setView);
  const [images, setImages] = useState<GeneratedImageRef[]>([]);
  const [nextBefore, setNextBefore] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<GeneratedImageRef | null>(null);

  const load = useCallback(async (before?: string) => {
    setStatus("loading");
    setError(null);
    try {
      const res = await listGeneratedImages(50, before);
      setImages((prev) => (before ? [...prev, ...res.images] : res.images));
      setNextBefore(res.next_before);
      setStatus("idle");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="flex h-full flex-col bg-neutral-950 text-neutral-100">
      <header className="flex items-center justify-between px-5 py-3">
        <h1 className="text-sm font-semibold tracking-wide text-neutral-200">
          ギャラリー
        </h1>
        <div className="flex items-center gap-3 text-xs text-neutral-500">
          <button
            type="button"
            onClick={() => setView("chat")}
            className="rounded-md px-2.5 py-1 text-xs text-neutral-400 transition hover:bg-neutral-900 hover:text-neutral-200"
          >
            チャットに戻る
          </button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-5 pb-5">
        {error && (
          <div className="mb-3 rounded-md bg-red-950/40 px-3 py-2 text-xs text-red-200">
            {error}
          </div>
        )}
        {images.length === 0 && status !== "loading" && (
          <div className="mt-10 text-center text-xs text-neutral-500">
            生成済みの画像はまだありません。
          </div>
        )}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
          {images.map((img) => (
            <button
              key={img.image_id}
              type="button"
              onClick={() => setSelected(img)}
              className="group relative aspect-square overflow-hidden rounded-lg bg-neutral-900 transition hover:ring-2 hover:ring-neutral-400"
            >
              <img
                src={img.image_url}
                alt={img.prompt_excerpt}
                loading="lazy"
                className="h-full w-full object-cover transition group-hover:opacity-90"
              />
            </button>
          ))}
        </div>
        {nextBefore && (
          <div className="mt-4 flex justify-center">
            <button
              type="button"
              onClick={() => void load(nextBefore)}
              disabled={status === "loading"}
              className="rounded-md bg-neutral-900 px-3 py-1.5 text-xs text-neutral-300 transition hover:bg-neutral-800 disabled:opacity-50"
            >
              {status === "loading" ? "読み込み中…" : "さらに読み込む"}
            </button>
          </div>
        )}
      </div>

      {selected && (
        <GalleryDetailPanel
          image={selected}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
