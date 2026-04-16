import type { ContentPart } from "@cocktail/api-types";

import { cn } from "../lib/cn";

interface Props {
  part: ContentPart;
}

export function MessagePart({ part }: Props): JSX.Element {
  switch (part.type) {
    case "text":
      return <p className="whitespace-pre-wrap leading-relaxed">{part.text}</p>;
    case "image":
      return (
        <figure className="mt-2">
          <img
            src={`/images/${part.image_id}.webp`}
            alt=""
            className="max-w-full rounded-lg border border-neutral-800"
            width={part.width ?? undefined}
            height={part.height ?? undefined}
            loading="lazy"
          />
        </figure>
      );
    case "tool_call":
      return (
        <div
          className={cn(
            "mt-2 rounded-md border px-3 py-2 text-xs",
            part.status === "running" && "border-blue-700/60 bg-blue-950/40 text-blue-200",
            part.status === "done" && "border-neutral-800 bg-neutral-900 text-neutral-400",
            part.status === "error" && "border-red-700/60 bg-red-950/40 text-red-200",
          )}
        >
          <div className="font-mono">
            <span className="opacity-70">tool:</span> {part.name}
            {part.status === "running" && <span className="ml-2 opacity-70">実行中…</span>}
          </div>
          {typeof part.args.prompt === "string" && (
            <div className="mt-1 opacity-80">→ {part.args.prompt}</div>
          )}
        </div>
      );
    case "tool_result":
      return (
        <div className="mt-2 rounded-md border border-neutral-800 bg-neutral-900 px-3 py-2 text-xs text-neutral-400">
          <span className="font-mono opacity-70">result:</span> {part.summary}
        </div>
      );
  }
}
