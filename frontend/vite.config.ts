import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies API calls to the FastAPI server on :8000.
// `npm run build` emits dist/, which the API serves at http://localhost:8000/.
const api = process.env.CODEFUSION_API ?? "http://localhost:8000";
const routes = ["/health", "/search", "/stats", "/index", "/versions", "/experiments", "/demo", "/graph", "/snippets"];

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: Object.fromEntries(routes.map((r) => [r, { target: api, changeOrigin: true }])) },
  build: { outDir: "dist", sourcemap: false },
});
