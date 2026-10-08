import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

const apiOrigin = process.env.LOCALROK_API_ORIGIN ?? "http://127.0.0.1:8080";

export default defineConfig({
  plugins: [tailwindcss()],
  resolve: {
    alias: [
      {
        find: /^fonts\/(KaTeX_.*)$/,
        replacement: fileURLToPath(new URL("./node_modules/katex/dist/fonts/", import.meta.url)) + "$1",
      },
    ],
  },
  build: {
    target: "esnext",
  },
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: apiOrigin,
        changeOrigin: true,
      },
      "/v1": {
        target: apiOrigin,
        changeOrigin: true,
      },
      "/figures": {
        target: apiOrigin,
        changeOrigin: true,
      },
    },
  },
});
