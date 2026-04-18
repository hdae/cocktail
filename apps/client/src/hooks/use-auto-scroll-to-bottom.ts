import { useCallback, useEffect, useRef, useState } from "react";

/**
 * ScrollArea の末尾追随スクロールを提供するフック。
 *
 * - `anchorRef` を ScrollArea 内の末尾に置くと、その `closest()` で viewport を
 *   逆引きする。Base UI の ScrollArea は function component で ref を素通ししない
 *   ため、DOM 経由で viewport を掴むのが一番確実。
 * - scroll event で「末尾付近にいるか」を ref（ホットパス判定用）と state
 *   （scroll-to-bottom ボタンの表示切替用）の両方へ同期させる。
 * - 自動追随の発火タイミングは呼び出し側が `followIfNeeded` を叩いて制御する。
 *   スムース/インスタントの使い分けはフック側ではなく呼び出し側の意図に任せる。
 */

const FOLLOW_THRESHOLD_PX = 96;
const VIEWPORT_SELECTOR = '[data-slot="scroll-area-viewport"]';

interface AutoScrollControls {
  anchorRef: React.RefObject<HTMLDivElement>;
  isAtBottom: boolean;
  scrollToBottom: (smooth?: boolean) => void;
  followIfNeeded: (smooth?: boolean) => void;
}

export function useAutoScrollToBottom(): AutoScrollControls {
  const anchorRef = useRef<HTMLDivElement>(null);
  const shouldFollowRef = useRef(true);
  const [isAtBottom, setIsAtBottom] = useState(true);

  const getViewport = useCallback(
    (): HTMLElement | null =>
      anchorRef.current?.closest<HTMLElement>(VIEWPORT_SELECTOR) ?? null,
    [],
  );

  const scrollToBottom = useCallback(
    (smooth = false): void => {
      const vp = getViewport();
      if (!vp) return;
      vp.scrollTo({
        top: vp.scrollHeight,
        behavior: smooth ? "smooth" : "auto",
      });
    },
    [getViewport],
  );

  const followIfNeeded = useCallback(
    (smooth = false): void => {
      if (!shouldFollowRef.current) return;
      scrollToBottom(smooth);
    },
    [scrollToBottom],
  );

  // scroll event で追随状態を同期。プログラム側の scrollTo でも event が飛ぶので、
  // ユーザ操作/プログラム操作どちらも同じロジックで判定される。
  useEffect(() => {
    const vp = getViewport();
    if (!vp) return;
    const update = (): void => {
      const near =
        vp.scrollHeight - vp.scrollTop - vp.clientHeight < FOLLOW_THRESHOLD_PX;
      shouldFollowRef.current = near;
      setIsAtBottom((prev) => (prev === near ? prev : near));
    };
    update();
    vp.addEventListener("scroll", update, { passive: true });
    return () => vp.removeEventListener("scroll", update);
  }, [getViewport]);

  // 画像に width/height 属性が無く aspect-ratio 予約が効かないケースの layout shift 保険。
  // 属性があれば通常は発動しない。
  useEffect(() => {
    const vp = getViewport();
    if (!vp) return;
    const onLoad = (ev: Event): void => {
      if (!(ev.target instanceof HTMLImageElement)) return;
      if (!shouldFollowRef.current) return;
      scrollToBottom(false);
    };
    vp.addEventListener("load", onLoad, true);
    return () => vp.removeEventListener("load", onLoad, true);
  }, [getViewport, scrollToBottom]);

  return { anchorRef, isAtBottom, scrollToBottom, followIfNeeded };
}
