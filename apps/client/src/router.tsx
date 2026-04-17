import {
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
  redirect,
} from "@tanstack/react-router";

import { AppShell } from "./components/AppShell";
import { ChatView } from "./components/ChatView";
import { GalleryView } from "./components/GalleryView";
import { HistoryView } from "./components/HistoryView";
import { getConversation } from "./lib/api";
import { useChatStore } from "./store/chat";

/**
 * `/` は未開始チャットへ、チャットは `conversationId` を URL に載せて識別する。
 * 既存会話のルートは loader で `GET /conversations/{id}` を呼び、store に注入する
 * ことでリロード / 直開きで履歴を復元できる。
 */
const rootRoute = createRootRoute({
  component: () => (
    <AppShell>
      <Outlet />
    </AppShell>
  ),
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  beforeLoad: () => {
    throw redirect({ to: "/conversations/new" });
  },
});

const newConversationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/conversations/new",
  component: () => <ChatView conversationId="new" />,
});

const existingConversationRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/conversations/$conversationId",
  loader: async ({ params }) => {
    // ストアが既にこの会話 ID を抱えている場合は hydrate しない。
    // 新規会話送信直後に `/conversations/new` → `/conversations/{uuid}` へ
    // navigate されたタイミングでは SSE がまだ走っており、ここで
    // `hydrateConversation` を呼ぶと abort controller を kill してしまう。
    const state = useChatStore.getState();
    if (state.conversationId === params.conversationId) {
      return null;
    }
    const detail = await getConversation(params.conversationId);
    useChatStore.getState().hydrateConversation(detail.id, detail.messages);
    return detail;
  },
  component: function ExistingConversationRoute() {
    const { conversationId } = existingConversationRoute.useParams();
    return <ChatView conversationId={conversationId} />;
  },
});

const galleryRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/gallery",
  component: () => <GalleryView />,
});

const historyRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/history",
  component: () => <HistoryView />,
});

const routeTree = rootRoute.addChildren([
  indexRoute,
  newConversationRoute,
  existingConversationRoute,
  galleryRoute,
  historyRoute,
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
