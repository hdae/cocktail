import type { ContentPart, ToolCallPart } from "@cocktail/api-types";

import { cn } from "../lib/utils";

interface Props {
  part: ContentPart;
}

export function MessagePart({ part }: Props): JSX.Element | null {
  switch (part.type) {
    case "text":
      return <p className="whitespace-pre-wrap leading-relaxed">{part.text}</p>;
    case "image": {
      const src = `/api/images/${part.image_id}.webp`;
      return (
        <figure className="mt-1 space-y-2">
          <a
            href={src}
            target="_blank"
            rel="noopener noreferrer"
            className="block cursor-zoom-in"
            title="画像を新しいタブで開く"
          >
            <img
              src={src}
              alt=""
              className="h-auto max-h-[400px] w-auto max-w-full rounded-lg transition hover:opacity-95"
              width={part.width ?? undefined}
              height={part.height ?? undefined}
            />
          </a>
          <div className="flex justify-end">
            <a
              href={src}
              download={`cocktail-${part.image_id}.webp`}
              className="inline-flex items-center gap-1.5 rounded-md bg-neutral-800/80 px-2.5 py-1 text-xs text-neutral-300 transition hover:bg-neutral-700 hover:text-neutral-100"
            >
              <DownloadIcon />
              ダウンロード
            </a>
          </div>
        </figure>
      );
    }
    case "tool_call":
      return <ToolCallView part={part} />;
    case "tool_result":
      // tool_result は ToolCallView 側でまとめて扱わないので、現状は非表示。
      return null;
  }
}

function ToolCallView({ part }: { part: ToolCallPart }): JSX.Element {
  if (part.status === "running" || part.status === "pending") {
    return (
      <div className="flex items-center gap-2 text-xs text-neutral-500">
        <Spinner />
        <span>画像を生成しています…</span>
      </div>
    );
  }

  if (part.status === "error") {
    return (
      <details className="rounded-md bg-red-950/30 px-3 py-2 text-xs text-red-200">
        <summary className="cursor-pointer select-none">
          画像生成でエラーが発生しました
        </summary>
        <ToolCallDetails args={part.args} />
      </details>
    );
  }

  return (
    <details className="group rounded-md bg-neutral-900/50 px-3 py-2 text-xs text-neutral-400">
      <summary className="cursor-pointer select-none text-neutral-500 transition hover:text-neutral-300">
        プロンプト詳細
      </summary>
      <ToolCallDetails args={part.args} />
    </details>
  );
}

function ToolCallDetails({ args }: { args: Record<string, unknown> }): JSX.Element {
  const entries = [
    ["positive", stringify(args.positive)],
    ["negative", stringify(args.negative)],
    ["aspect_ratio", stringify(args.aspect_ratio)],
    ["cfg_preset", stringify(args.cfg_preset)],
    ["seed", stringify(args.seed)],
    ["resolution", formatResolution(args.width, args.height)],
  ].filter(([, v]) => v !== null) as [string, string][];

  return (
    <dl className="mt-2 space-y-1.5 text-[11px] leading-relaxed">
      {entries.map(([k, v]) => (
        <div key={k} className="flex gap-2">
          <dt className="w-24 shrink-0 font-mono text-neutral-500">{k}</dt>
          <dd
            className={cn(
              "min-w-0 flex-1 break-words text-neutral-300",
              (k === "positive" || k === "negative") && "font-mono",
            )}
          >
            {v}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function stringify(v: unknown): string | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "string") return v.length > 0 ? v : null;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return JSON.stringify(v);
}

function formatResolution(w: unknown, h: unknown): string | null {
  if (typeof w === "number" && typeof h === "number") return `${w} × ${h}`;
  return null;
}

function Spinner(): JSX.Element {
  return (
    <span
      className="inline-block h-3 w-3 animate-spin rounded-full border border-neutral-600 border-t-transparent"
      aria-hidden="true"
    />
  );
}

function DownloadIcon(): JSX.Element {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      className="h-3.5 w-3.5"
      aria-hidden="true"
    >
      <path d="M10 2.5a.75.75 0 0 1 .75.75v8.19l2.72-2.72a.75.75 0 1 1 1.06 1.06l-4 4a.75.75 0 0 1-1.06 0l-4-4a.75.75 0 1 1 1.06-1.06l2.72 2.72V3.25A.75.75 0 0 1 10 2.5Z" />
      <path d="M3.75 14a.75.75 0 0 1 .75.75v1.5c0 .138.112.25.25.25h10.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 15.25 18H4.75A1.75 1.75 0 0 1 3 16.25v-1.5a.75.75 0 0 1 .75-.75Z" />
    </svg>
  );
}
