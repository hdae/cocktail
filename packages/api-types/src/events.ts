import { z } from "zod";

import { MessageSchema } from "./messages.js";

/**
 * `POST /chat` が SSE で流すイベント。サーバ側 `schemas/events.py` と完全対応。
 * `type` フィールドは SSE の `event:` 名とも一致するので、ストリーム側でも
 * 同じ string で分岐してよい。
 */

export const ConversationEventSchema = z
  .object({
    type: z.literal("conversation"),
    conversation_id: z.string(),
  })
  .strict();

export const UserSavedEventSchema = z
  .object({
    type: z.literal("user_saved"),
    message: MessageSchema,
  })
  .strict();

export const AssistantStartEventSchema = z
  .object({
    type: z.literal("assistant_start"),
    message_id: z.string(),
  })
  .strict();

export const ToolCallStartEventSchema = z
  .object({
    type: z.literal("tool_call_start"),
    call_id: z.string(),
    name: z.string(),
    args: z.record(z.unknown()),
  })
  .strict();

export const ToolCallEndEventSchema = z
  .object({
    type: z.literal("tool_call_end"),
    call_id: z.string(),
    status: z.enum(["done", "error"]),
    summary: z.string(),
    data: z.record(z.unknown()),
  })
  .strict();

export const ImageReadyEventSchema = z
  .object({
    type: z.literal("image_ready"),
    call_id: z.string(),
    image_id: z.string(),
    image_url: z.string(),
    mime: z.string(),
    width: z.number().int(),
    height: z.number().int(),
  })
  .strict();

export const AssistantEndEventSchema = z
  .object({
    type: z.literal("assistant_end"),
    message: MessageSchema,
  })
  .strict();

export const ErrorEventSchema = z
  .object({
    type: z.literal("error"),
    code: z.string(),
    message: z.string(),
  })
  .strict();

export const DoneEventSchema = z
  .object({
    type: z.literal("done"),
  })
  .strict();

export const SseEventSchema = z.discriminatedUnion("type", [
  ConversationEventSchema,
  UserSavedEventSchema,
  AssistantStartEventSchema,
  ToolCallStartEventSchema,
  ToolCallEndEventSchema,
  ImageReadyEventSchema,
  AssistantEndEventSchema,
  ErrorEventSchema,
  DoneEventSchema,
]);
export type SseEvent = z.infer<typeof SseEventSchema>;

export type ConversationEvent = z.infer<typeof ConversationEventSchema>;
export type UserSavedEvent = z.infer<typeof UserSavedEventSchema>;
export type AssistantStartEvent = z.infer<typeof AssistantStartEventSchema>;
export type ToolCallStartEvent = z.infer<typeof ToolCallStartEventSchema>;
export type ToolCallEndEvent = z.infer<typeof ToolCallEndEventSchema>;
export type ImageReadyEvent = z.infer<typeof ImageReadyEventSchema>;
export type AssistantEndEvent = z.infer<typeof AssistantEndEventSchema>;
export type ErrorEvent = z.infer<typeof ErrorEventSchema>;
export type DoneEvent = z.infer<typeof DoneEventSchema>;
