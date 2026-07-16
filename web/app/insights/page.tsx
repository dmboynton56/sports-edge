import Link from "next/link";
import { BarChart3, Brain, ChevronRight } from "lucide-react";

import { PageHeader } from "@/components/dashboard/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const posts = [
  {
    href: "/insights/grading-roundup-2026",
    title: "2026 Grading & Backtest Roundup",
    description:
      "Live graded results, persisted backtest evidence, calibration coverage, and production readiness by sport.",
    icon: BarChart3,
  },
  {
    href: "/insights/mlb-hr-pytorch",
    title: "MLB HR PyTorch Experiment",
    description:
      "Baseline random-forest home-run metrics, the GPU training plan, and the before/after comparison slot.",
    icon: Brain,
  },
];

export default function InsightsPage() {
  return (
    <div>
      <PageHeader
        title="Insights"
        description="Research notes, model post-mortems, and publish-ready experiment writeups."
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {posts.map((post) => {
          const Icon = post.icon;
          return (
            <Card key={post.href}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Icon className="size-5 text-accent" />
                  {post.title}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="mb-4 text-sm leading-6 text-muted-foreground">{post.description}</p>
                <Link className="inline-flex items-center gap-1 text-sm font-medium text-accent" href={post.href}>
                  Open insight
                  <ChevronRight className="size-4" />
                </Link>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
