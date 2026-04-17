import { z } from "zod";

import { GeneratedImageRefSchema } from "./images.js";
import { MessageSchema } from "./messages.js";

/**
 * `apps/server/src/cocktail_server/schemas/conversations.py` の `ConversationDetail` を
 * ミラーする。URL 直開き / リロード時に会話を復元する `GET /conversations/{id}` の応答。
 */
export const ConversationDetailSchema = z
  .object({
    id: z.string().min(1).max(128),
    created_at: z.string(),
    updated_at: z.string(),
    messages: z.array(MessageSchema),
    generated_images: z.array(GeneratedImageRefSchema),
  })
  .strict();

export type ConversationDetail = z.infer<typeof ConversationDetailSchema>;

/**
 * `apps/server/src/cocktail_server/schemas/conversations.py` の `ConversationSummary` を
 * ミラーする。左メニューの履歴一覧向け軽量サマリ。
 */
export const ConversationSummarySchema = z
  .object({
    id: z.string().min(1).max(128),
    title: z.string().min(1).max(80),
    created_at: z.string(),
    updated_at: z.string(),
    message_count: z.number().int().min(0),
  })
  .strict();

export type ConversationSummary = z.infer<typeof ConversationSummarySchema>;
