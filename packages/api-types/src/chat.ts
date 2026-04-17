import { z } from "zod";

import { TextPartSchema, UserContentPartSchema } from "./messages.js";

/** `POST /chat` のリクエスト。text パートが少なくとも 1 つ含まれることを要求。 */
export const ChatRequestSchema = z
  .object({
    conversation_id: z.string().max(128).nullable().optional(),
    parts: z.array(UserContentPartSchema).min(1).max(32),
    parent_id: z.string().max(128).nullable().optional(),
  })
  .strict()
  .refine(
    (req) => req.parts.some((p) => TextPartSchema.safeParse(p).success),
    { message: "parts must contain at least one text part", path: ["parts"] },
  );

export type ChatRequest = z.infer<typeof ChatRequestSchema>;

/**
 * `POST /chat` の応答。SSE は別接続 `GET /chat/turns/{turn_id}/events` で購読する。
 */
export const ChatStartResponseSchema = z
  .object({
    conversation_id: z.string(),
    turn_id: z.string(),
  })
  .strict();

export type ChatStartResponse = z.infer<typeof ChatStartResponseSchema>;
