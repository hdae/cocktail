import { ChatView } from "./components/ChatView";

export function App(): JSX.Element {
  return (
    <main className="mx-auto flex h-full max-w-3xl flex-col">
      <ChatView />
    </main>
  );
}
