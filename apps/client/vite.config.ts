import path from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      // サーバは 8000 で動いている。SSE は WebSocket と違って通常の HTTP なのでそのままプロキシ可能。
      "/chat": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/images": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/generate": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/health": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
