import { EmptyState } from "@/components/dashboard/EmptyState";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { Badge } from "@/components/ui/badge";

export default function NhlMarketsPage() {
  return (
    <div>
      <PageHeader
        title="NHL markets"
        description="A placeholder in the league hierarchy — nothing is modelled here yet."
      />
      <EmptyState
        title="Nothing modelled for hockey yet"
        description="NHL holds a slot in the sport hierarchy so ingest and routing are ready, but no model writes to this board. Every other league is live or seasonal."
        detail={<Badge variant="outline">Scaffold only</Badge>}
      />
    </div>
  );
}
