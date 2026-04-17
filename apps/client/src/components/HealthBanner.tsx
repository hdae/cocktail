import { useEffect, useState } from "react";

import { type HealthResponse, HealthResponseSchema } from "@cocktail/api-types";

type FetchState =
  | { kind: "probing" }
  | { kind: "unreachable" }
  | { kind: "ok"; data: HealthResponse };

const POLL_INTERVAL_MS = 2000;

/** `/health` を polling し、ready になるまで回し続ける。ready 後は 10s 間隔に落とす。 */
function useHealth(): FetchState {
  const [state, setState] = useState<FetchState>({ kind: "probing" });

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async (): Promise<void> => {
      try {
        const res = await fetch("/health", { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json: unknown = await res.json();
        const data = HealthResponseSchema.parse(json);
        if (cancelled) return;
        setState({ kind: "ok", data });
        const delay = data.startup.state === "ready" ? 10_000 : POLL_INTERVAL_MS;
        timer = setTimeout(tick, delay);
      } catch {
        if (cancelled) return;
        setState({ kind: "unreachable" });
        timer = setTimeout(tick, POLL_INTERVAL_MS);
      }
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  return state;
}

export function HealthBanner(): JSX.Element | null {
  const state = useHealth();

  if (state.kind === "probing") {
    return <Banner tone="muted" label="サーバ状態を確認中…" spinner />;
  }

  if (state.kind === "unreachable") {
    return <Banner tone="warn" label="サーバに接続できません（再試行中）" spinner />;
  }

  const { startup } = state.data;
  if (startup.state === "ready") return null;

  if (startup.state === "downloading" || startup.state === "loading") {
    return (
      <Banner
        tone="muted"
        label="モデルの準備中…（初回やモデル変更時は数分〜十数分かかります）"
        spinner
      />
    );
  }

  // error
  const message = startup.error ?? "原因不明のエラー";
  return <Banner tone="error" label={`起動に失敗しました: ${message}`} />;
}

interface BannerProps {
  tone: "muted" | "warn" | "error";
  label: string;
  spinner?: boolean;
}

function Banner({ tone, label, spinner }: BannerProps): JSX.Element {
  const toneClass =
    tone === "error"
      ? "bg-red-950/50 text-red-200"
      : tone === "warn"
        ? "bg-amber-950/40 text-amber-200"
        : "bg-neutral-900/70 text-neutral-300";
  return (
    <div
      className={`flex items-center gap-2 rounded-md px-3 py-2 text-xs ${toneClass}`}
    >
      {spinner && <Spinner />}
      <span>{label}</span>
    </div>
  );
}

function Spinner(): JSX.Element {
  return (
    <span
      className="inline-block h-3 w-3 animate-spin rounded-full border border-current border-t-transparent"
      aria-hidden="true"
    />
  );
}
