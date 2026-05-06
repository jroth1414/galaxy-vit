import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Galaxy-ViT frontend.
// Dev mode: vite serves on :5173 with HMR; FastAPI runs separately on :8000
// and the proxy below forwards /api and /static so the SPA can call the
// real backend without CORS gymnastics.
// Prod mode: `vite build` -> frontend/dist/ which FastAPI mounts at / on
// :7860 (the HF Spaces port).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/static': 'http://localhost:8000',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
