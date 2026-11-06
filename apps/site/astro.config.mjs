import { defineConfig } from "astro/config";

// Static marketing site for minastir.io. The product itself (apps/web + the Rust server) runs on
// the user's own machine; this site only explains it.
export default defineConfig({
  site: "https://minastir.io",
  // three.js is one ~500 kB (≈125 kB gzipped) chunk, loaded after first paint.
  vite: { build: { chunkSizeWarningLimit: 700 } },
});
