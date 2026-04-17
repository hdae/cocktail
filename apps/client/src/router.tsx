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

const routeTree = rootRoute.addChildren([
  indexRoute,
  newConversationRoute,
  existingConversationRoute,
  galleryRoute,
]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
