import { useCallback, useRef, useState, type DragEvent, type KeyboardEvent } from "react";

import type { ImagePart, UserContentPart } from "@cocktail/api-types";

import { uploadImage } from "../lib/api";
import { cn } from "../lib/cn";

interface Props {
  disabled?: boolean;
  onSend: (parts: UserContentPart[]) => void;
}

interface Attachment {
  id: string;
  url: string;
  part: ImagePart;
}

export function ComposerInput({ disabled, onSend }: Props): JSX.Element {
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const addFiles = useCallback(async (files: FileList | File[]) => {
    const images = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (images.length === 0) return;
    setUploading(true);
    try {
      for (const file of images) {
        const res = await uploadImage(file);
        setAttachments((prev) => [
          ...prev,
          {
            id: res.image_id,
            url: res.image_url,
            part: {
              type: "image",
              image_id: res.image_id,
              mime: res.mime,
              width: res.width,
              height: res.height,
            },
          },
        ]);
      }
    } catch (err) {
      console.error(err);
      alert(`画像アップロードに失敗: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setUploading(false);
    }
  }, []);

  const handlePaste = useCallback(
    (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
      const files = Array.from(e.clipboardData.files);
      if (files.length > 0) {
        e.preventDefault();
        void addFiles(files);
      }
    },
    [addFiles],
  );

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      void addFiles(e.dataTransfer.files);
    },
    [addFiles],
  );

  const send = useCallback(() => {
    const body = text.trim();
    if (!body && attachments.length === 0) return;
    const parts: UserContentPart[] = [];
    for (const att of attachments) parts.push(att.part);
    if (body) parts.push({ type: "text", text: body });
    // text が無いとサーバ側 422 になる。attachments だけのときは一応プロンプトで補う。
    if (!body && attachments.length > 0) {
      parts.push({ type: "text", text: "この画像を参考に生成してください。" });
    }
    onSend(parts);
    setText("");
    setAttachments([]);
  }, [text, attachments, onSend]);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        send();
      }
    },
    [send],
  );

  return (
    <div
      className={cn(
        "border-t border-neutral-800 bg-neutral-950 p-3",
        dragOver && "bg-blue-950/30",
      )}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      {attachments.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2">
          {attachments.map((att) => (
            <div key={att.id} className="relative">
              <img
                src={att.url}
                alt=""
                className="h-16 w-16 rounded border border-neutral-700 object-cover"
              />
              <button
                type="button"
                onClick={() =>
                  setAttachments((prev) => prev.filter((a) => a.id !== att.id))
                }
                className="absolute -right-2 -top-2 rounded-full bg-neutral-800 px-1.5 text-xs text-white hover:bg-neutral-700"
                aria-label="削除"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      <div className="flex items-end gap-2">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || uploading}
          className="shrink-0 rounded-md border border-neutral-700 px-3 py-2 text-sm text-neutral-300 hover:bg-neutral-900 disabled:opacity-50"
          aria-label="画像を添付"
        >
          📎
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files) void addFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          onPaste={handlePaste}
          placeholder="日本語で描きたいものを書いてください (Cmd/Ctrl + Enter で送信)"
          rows={2}
          className="flex-1 resize-none rounded-md border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 placeholder:text-neutral-500 focus:border-blue-600 focus:outline-none"
          disabled={disabled}
        />
        <button
          type="button"
          onClick={send}
          disabled={disabled || uploading || (!text.trim() && attachments.length === 0)}
          className="shrink-0 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
        >
          送信
        </button>
      </div>
      {uploading && (
        <p className="mt-1 text-xs text-neutral-500">画像をアップロード中…</p>
      )}
    </div>
  );
}
