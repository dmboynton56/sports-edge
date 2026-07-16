import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      {
        source: "/pga",
        destination: "/markets/pga",
        permanent: false,
      },
      {
        source: "/cbb",
        destination: "/markets/cbb",
        permanent: false,
      },
      {
        source: "/markets/mlb-home-runs",
        destination: "/markets/mlb/home-runs",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
