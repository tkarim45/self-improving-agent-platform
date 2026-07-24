import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Minimal self-contained server for the Docker image (.next/standalone/server.js).
  output: "standalone",
  // Proxy API calls to the FastAPI backend so the browser talks same-origin.
  // Dev: BACKEND_URL defaults to localhost:8000. Docker: compose sets it to the backend
  // service. The rewrite runs server-side, so the browser never sees the backend host.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.BACKEND_URL ?? "http://127.0.0.1:8000"}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
