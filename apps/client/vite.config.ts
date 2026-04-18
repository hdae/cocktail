import path from "node:path";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      // サーバ API はすべて `/api` 配下。ここだけを 8000 にプロキシし、
      // それ以外 (`/conversations/xxx` などの SPA ルート) は Vite が index.html を
      // 返すため、リロード / 直接 URL 踏みでも SPA 側が描画を担う。
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
