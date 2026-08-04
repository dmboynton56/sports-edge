import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { AppShell } from "@/components/dashboard/AppShell";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Sports Edge | Sports Analytics & Fantasy Football",
  description: "Sports betting model performance, NFL fantasy projections, and configurable lineup planning.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        suppressHydrationWarning
        className={`${inter.className} min-h-screen bg-background text-foreground selection:bg-accent selection:text-accent-foreground`}
      >
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
