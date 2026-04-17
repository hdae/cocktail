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
