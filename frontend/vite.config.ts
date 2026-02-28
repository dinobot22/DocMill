import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/health': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/models': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/engines': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/infer': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/history': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/files': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
})