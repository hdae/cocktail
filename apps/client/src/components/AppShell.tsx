import type { ReactNode } from "react";

import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";

import { AppSidebar } from "./AppSidebar";

interface Props {
  children: ReactNode;
}

/**
 * ルート直下のシェル。左に `AppSidebar`（PC では常時表示、モバイルでは Sheet）、
 * 右に現在のルートコンポーネントを `SidebarInset` で並べる。
 */
export function AppShell({ children }: Props): JSX.Element {
  return (
    <SidebarProvider defaultOpen>
      <AppSidebar />
      <SidebarInset className="flex h-svh flex-col overflow-hidden bg-background">
        {children}
      </SidebarInset>
    </SidebarProvider>
  );
}
