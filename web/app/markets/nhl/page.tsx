import { PageHeader } from "@/components/dashboard/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function NhlMarketsPage() {
  return (
    <div>
      <PageHeader
        title="NHL Markets"
        description="NHL model coverage is reserved in the sport hierarchy."
      />
      <Card className="border-dashed bg-muted/20 text-muted-foreground">
        <CardHeader>
          <div className="flex items-center justify-between gap-3">
            <CardTitle className="text-base text-foreground">NHL model board</CardTitle>
            <Badge variant="secondary" className="text-muted-foreground">
              scaffold
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="text-sm">
          No NHL models are wired to the dashboard yet.
        </CardContent>
      </Card>
    </div>
  );
}
