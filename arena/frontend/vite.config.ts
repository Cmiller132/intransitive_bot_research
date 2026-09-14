import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// `npm run dev` proxies API calls to a locally running arena backend
// (uvicorn arena.api.app:app on :8100); override the target with VITE_API_PROXY.
export default defineConfig(() => {
  const target = process.env.VITE_API_PROXY || 'http://127.0.0.1:8100'
  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': target,
        '/healthz': target,
      },
    },
  }
})
