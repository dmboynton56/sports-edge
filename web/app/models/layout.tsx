import type { ReactNode } from "react";

import { ModelsNav } from "@/components/models/ModelsNav";

export default function ModelsLayout({ children }: { children: ReactNode }) {
  return (
    <>
      <ModelsNav />
      {children}
    </>
  );
}
