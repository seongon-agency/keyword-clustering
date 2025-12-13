import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    let backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
    // Auto-add https:// if protocol is missing
    if (backendUrl && !backendUrl.startsWith("http://") && !backendUrl.startsWith("https://")) {
      backendUrl = `https://${backendUrl}`;
    }
    return [
      {
        source: "/api/python/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
