import { ChatView } from "./components/ChatView";
import { GalleryView } from "./components/GalleryView";
import { useChatStore } from "./store/chat";

export function App(): JSX.Element {
  const view = useChatStore((s) => s.view);
  return (
    <main className="mx-auto flex h-full max-w-5xl flex-col">
      {view === "gallery" ? <GalleryView /> : <ChatView />}
    </main>
  );
}
