import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// アシスタントの会話テキストを GitHub-flavored Markdown で描画する。react-markdown は
// 既定で生 HTML を描画しない（XSS 安全）。@tailwindcss/typography は未導入なので、
// 各要素に最小限の Tailwind を割り当ててダークテーマに馴染ませる。
const components: Components = {
  p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="my-2 list-disc space-y-0.5 pl-5">{children}</ul>,
  ol: ({ children }) => <ol className="my-2 list-decimal space-y-0.5 pl-5">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-neutral-100">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-sky-400 underline underline-offset-2 hover:text-sky-300"
    >
      {children}
    </a>
  ),
  h1: ({ children }) => <h1 className="my-2 text-lg font-semibold">{children}</h1>,
  h2: ({ children }) => <h2 className="my-2 text-base font-semibold">{children}</h2>,
  h3: ({ children }) => <h3 className="my-2 text-sm font-semibold">{children}</h3>,
  code: ({ children }) => (
    <code className="rounded bg-neutral-800 px-1 py-0.5 font-mono text-[0.85em]">{children}</code>
  ),
  pre: ({ children }) => (
    <pre className="my-2 overflow-x-auto rounded-lg bg-neutral-900 p-3 text-[0.85em] [&>code]:bg-transparent [&>code]:p-0">
      {children}
    </pre>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-neutral-700 pl-3 text-neutral-400">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-3 border-neutral-800" />,
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-neutral-800 px-2 py-1 text-left font-semibold">{children}</th>
  ),
  td: ({ children }) => <td className="border border-neutral-800 px-2 py-1">{children}</td>,
};

export function Markdown({ text }: { text: string }): JSX.Element {
  return (
    <div className="leading-relaxed">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
