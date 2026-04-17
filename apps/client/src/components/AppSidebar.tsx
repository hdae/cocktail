import { Link, useRouterState } from "@tanstack/react-router";
import { ImageIcon, MessageSquarePlus } from "lucide-react";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";

type NavItem = {
  label: string;
  to: string;
  icon: typeof MessageSquarePlus;
  /** 現在パスが前方一致するかの判定に使う。`to` と違うケースだけ指定する。 */
  matchPrefix?: string;
};

const NAV_ITEMS: readonly NavItem[] = [
  {
    label: "新規チャット",
    to: "/conversations/new",
    icon: MessageSquarePlus,
    matchPrefix: "/conversations",
  },
  {
    label: "ギャラリー",
    to: "/gallery",
    icon: ImageIcon,
  },
];

export function AppSidebar(): JSX.Element {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <Sidebar collapsible="offcanvas">
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Cocktail</SidebarGroupLabel>
          <SidebarMenu>
            {NAV_ITEMS.map((item) => {
              const prefix = item.matchPrefix ?? item.to;
              const isActive = pathname === item.to || pathname.startsWith(`${prefix}/`);
              const Icon = item.icon;
              return (
                <SidebarMenuItem key={item.to}>
                  <SidebarMenuButton asChild isActive={isActive} tooltip={item.label}>
                    <Link to={item.to}>
                      <Icon />
                      <span>{item.label}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              );
            })}
          </SidebarMenu>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
