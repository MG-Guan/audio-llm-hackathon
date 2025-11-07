// vite.config.ts
import { defineConfig } from 'vite';

export default defineConfig({
  // ... other Vite configurations
  server: {
    allowedHosts: [
      '*.ngrok-free.app', // Adjust based on your ngrok domain
      'localhost',
      '127.0.0.1'
    ],
  },
});