import { ChatView } from "./components/ChatView";

export function App(): JSX.Element {
  return (
    <main className="mx-auto flex h-full max-w-5xl flex-col">
      <ChatView />
    </main>
  );
}
