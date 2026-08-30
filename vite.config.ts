import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // used when Vite runs standalone; in middleware mode server.ts owns the port
    port: 3000,
    strictPort: false,
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
