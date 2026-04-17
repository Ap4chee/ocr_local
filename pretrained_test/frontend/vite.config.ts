import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const crossOriginIsolation = {
  name: "cross-origin-isolation",
  configureServer(server: any) {
    server.middlewares.use((_req: any, res: any, next: any) => {
      res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
      next();
    });
  },
  configurePreviewServer(server: any) {
    server.middlewares.use((_req: any, res: any, next: any) => {
      res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
      next();
    });
  },
};

export default defineConfig({
  plugins: [react(), crossOriginIsolation],
  server: { port: 5173 },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  worker: { format: "es" },
});
