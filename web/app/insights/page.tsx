import { ChannelCard } from "@/components/dashboard/ChannelCard";
import { PageHeader } from "@/components/dashboard/PageHeader";

const posts = [
  {
    href: "/insights/grading-roundup-2026",
    title: "2026 Grading & Backtest Roundup",
    description:
      "Live graded results, persisted backtest evidence, calibration coverage, and production readiness by sport.",
  },
  {
    href: "/insights/mlb-hr-pytorch",
    title: "MLB HR PyTorch Experiment",
    description:
      "Baseline random-forest home-run metrics, the GPU training plan, and the before/after comparison slot.",
  },
];

export default function InsightsPage() {
  return (
    <div>
      <PageHeader
        title="Insights"
        description="Post-mortems on what the models got wrong, and notes on what changed as a result."
      />

      <div className="grid gap-3 md:grid-cols-2">
        {posts.map((post) => (
          <ChannelCard
            key={post.href}
            href={post.href}
            title={post.title}
            description={post.description}
            cta="Read the write-up"
          />
        ))}
      </div>
    </div>
  );
}
