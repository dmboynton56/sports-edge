import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      {
        source: "/nba/:gameId",
        destination: "/markets/nba/:gameId",
        permanent: false,
      },
      {
        source: "/nfl/:gameId",
        destination: "/markets/nfl/:gameId",
        permanent: false,
      },
      {
        source: "/nba",
        destination: "/markets?sport=NBA&market=spread",
        permanent: false,
      },
      {
        source: "/nfl",
        destination: "/markets?sport=NFL&market=spread",
        permanent: false,
      },
      {
        source: "/performance/:path+",
        destination: "/models/performance/:path+",
        permanent: false,
      },
      {
        source: "/performance",
        destination: "/models/performance",
        permanent: false,
      },
      {
        source: "/results",
        destination: "/models/results",
        permanent: false,
      },
      {
        source: "/insights/:path+",
        destination: "/models/insights/:path+",
        permanent: false,
      },
      {
        source: "/insights",
        destination: "/models/insights",
        permanent: false,
      },
      {
        source: "/data-quality",
        destination: "/models/data-quality",
        permanent: false,
      },
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
