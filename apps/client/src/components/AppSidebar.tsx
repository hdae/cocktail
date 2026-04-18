import { Link, useRouterState } from "@tanstack/react-router";
import { History, ImageIcon, MessageSquarePlus } from "lucide-react";

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

/**
 * チャット履歴を左メニューに出すかの切替。
 * 共有モードでは有効、個人利用モードでは隠す想定。
 */
const SHOW_CHAT_HISTORY = import.meta.env.VITE_SHOW_CHAT_HISTORY === "true";

const BASE_NAV_ITEMS: readonly NavItem[] = [
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

const HISTORY_NAV_ITEM: NavItem = {
  label: "チャット履歴",
  to: "/history",
  icon: History,
};

const NAV_ITEMS: readonly NavItem[] = SHOW_CHAT_HISTORY
  ? [...BASE_NAV_ITEMS, HISTORY_NAV_ITEM]
  : BASE_NAV_ITEMS;

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
                  <SidebarMenuButton
                    render={<Link to={item.to} />}
                    isActive={isActive}
                    tooltip={item.label}
                  >
                    <Icon />
                    <span>{item.label}</span>
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
