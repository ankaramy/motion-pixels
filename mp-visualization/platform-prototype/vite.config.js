import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // GitHub Pages project site: served from https://ankaramy.github.io/motion-pixels/
  base: '/motion-pixels/',
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,   // fail instead of silently moving to 5174, 5175, etc.
    host: true,         // also reachable via local-network IP (e.g. 192.168.x.x)
    open: true,         // auto-opens the browser on npm run dev
  },
  preview: {
    port: 4173,
    strictPort: true,
    host: true,
    open: true,
  },
})
