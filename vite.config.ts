import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Extra hostnames for a dev preview behind a proxy, comma separated.
 * Left unset in the repository so a plain clone carries no environment-specific
 * configuration; the build itself is fully static and uses a relative base so it
 * can be served from any GitHub Pages project path.
 */
const previewHosts = (process.env.PREVIEW_HOSTS ?? "")
  .split(",")
  .map((h) => h.trim())
  .filter(Boolean);

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    host: "0.0.0.0",
    port: 3003,
    strictPort: true,
    ...(previewHosts.length
      ? { allowedHosts: previewHosts, hmr: { clientPort: 443, protocol: "wss" as const } }
      : {}),
  },
  build: { target: "es2020", assetsInlineLimit: 0 },
});
