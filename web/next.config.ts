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
      {
        source: "/performance",
        destination: "/record",
        permanent: false,
      },
      {
        source: "/performance/:sport",
        destination: "/record",
        permanent: false,
      },
      {
        source: "/results",
        destination: "/record",
        permanent: false,
      },
      {
        source: "/insights",
        destination: "/record",
        permanent: false,
      },
      {
        source: "/insights/:slug*",
        destination: "/record",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
