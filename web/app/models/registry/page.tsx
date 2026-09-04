import { PageHeader } from "@/components/dashboard/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { getEvaluationsBundle } from "@/lib/data/evaluations";
import { formatDateTime, formatNumber, formatPct } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function ModelRegistryPage() {
  const bundle = await getEvaluationsBundle();

  return (
    <div>
      <PageHeader
        title="Model Registry"
        description="Production model versions, evaluation evidence, and strategy backtests."
        meta={bundle.generatedAt}
      />

      {bundle.gaps.length ? (
        <div className="mb-4 flex flex-wrap gap-2">
          {bundle.gaps.map((gap) => (
            <Badge key={gap} variant="missing">
              {gap}
            </Badge>
          ))}
        </div>
      ) : null}

      <Card className="mb-4">
        <CardHeader>
          <CardTitle>Active Registry</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>League</TableHead>
                <TableHead>Version</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Notes</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {bundle.registry.map((entry) => (
                <TableRow key={`${entry.league}-${entry.modelVersion}`}>
                  <TableCell>{entry.league}</TableCell>
                  <TableCell>{entry.modelVersion}</TableCell>
                  <TableCell>
                    <Badge variant={entry.status === "production" ? "accent" : "outline"}>
                      {entry.status}
                    </Badge>
                  </TableCell>
                  <TableCell>{entry.notes}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="grid gap-4 [&>*]:min-w-0 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Evaluation Runs</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>League</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Eval</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {bundle.evaluations.slice(0, 12).map((row) => (
                  <TableRow key={row.id}>
                    <TableCell>{row.league}</TableCell>
                    <TableCell>
                      {row.modelName} {row.modelVersion}
                    </TableCell>
                    <TableCell>{row.evaluationName}</TableCell>
                    <TableCell>{row.status}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Strategy Backtests</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>League</TableHead>
                  <TableHead>Strategy</TableHead>
                  <TableHead>Sample</TableHead>
                  <TableHead>ROI</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {bundle.strategies.slice(0, 12).map((row) => (
                  <TableRow key={row.id}>
                    <TableCell>{row.league}</TableCell>
                    <TableCell>{row.strategyId}</TableCell>
                    <TableCell>{formatNumber(row.sampleSize)}</TableCell>
                    <TableCell>{formatPct(row.roi)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      <p className="mt-4 text-xs text-muted-foreground">
        Updated {formatDateTime(bundle.generatedAt)}
      </p>
    </div>
  );
}
