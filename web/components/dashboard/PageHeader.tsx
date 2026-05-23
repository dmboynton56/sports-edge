import { Badge } from "@/components/ui/badge";

export function PageHeader({
  title,
  description,
  meta,
}: {
  title: string;
  description: string;
  meta?: string | null;
}) {
  return (
    <div className="mb-5 flex flex-col gap-3 border-b border-border pb-5 md:flex-row md:items-end md:justify-between">
      <div>
        <h1 className="text-2xl font-semibold tracking-normal sm:text-3xl">{title}</h1>
        <p className="mt-2 max-w-3xl text-sm text-muted-foreground">{description}</p>
      </div>
      {meta ? (
        <Badge variant="outline" className="w-fit">
          Updated {meta}
        </Badge>
      ) : null}
    </div>
  );
}
