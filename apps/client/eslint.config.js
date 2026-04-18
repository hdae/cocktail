import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    // shadcn/ui からコピペしたコンポーネントとフックは「所有する第三者コード」扱いで
    // lint 対象外にする。自前で触るコード（ChatView, router, store, lib 等）だけを
    // gate する方針。
    ignores: [
      "dist",
      "node_modules",
      "src/components/ui/**",
      "src/hooks/use-mobile.tsx",
    ],
  },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      // `useEffect(() => void load(), [load])` は初回フェッチの定番パターンで、
      // load 内の setState が間接的に呼ばれるのを新ルールが誤検出する。
      // TanStack Query 等を入れたら削除する想定。
      "react-hooks/set-state-in-effect": "off",
    },
  },
);
