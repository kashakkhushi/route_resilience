import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  build: {
    rollupOptions: {
      output: {
        // Split the heavy, rarely-changing libraries out of the app chunk so a
        // code change does not force a re-download of the charting and mapping
        // runtimes. Recharts alone is roughly half the bundle.
        manualChunks: {
          react: ["react", "react-dom"],
          charts: ["recharts"],
          icons: ["lucide-react"],
        },
      },
    },
    chunkSizeWarningLimit: 700,
  },
  server: {
    port: 5173,
    // The API base URL is never baked into the bundle; the dev server proxies
    // /api so no origin or key ever ships to the browser.
    proxy: { "/api": { target: "http://localhost:8000", changeOrigin: true } },
  },
});
