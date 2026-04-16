import { z } from "zod";

/**
 * `apps/server/src/cocktail_server/schemas/messages.py` と同じ判別 union を
 * クライアント側 Zod で再現する。サーバとキー名・制約をそろえる責務があるので
 * 変更するときは両側をセットで更新すること。
 */

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

export const RoleSchema = z.enum(["user", "assistant", "tool", "system"]);
export type Role = z.infer<typeof RoleSchema>;

export const ToolCallStatusSchema = z.enum(["pending", "running", "done", "error"]);
export type ToolCallStatus = z.infer<typeof ToolCallStatusSchema>;

export const TextPartSchema = z
  .object({
    type: z.literal("text"),
    text: z.string().min(1).max(10_000),
  })
  .strict();
export type TextPart = z.infer<typeof TextPartSchema>;

export const ImagePartSchema = z
  .object({
    type: z.literal("image"),
    image_id: z.string().regex(UUID_PATTERN),
    mime: z.string().min(1).max(64),
    width: z.number().int().min(1).max(8192).nullable().optional(),
    height: z.number().int().min(1).max(8192).nullable().optional(),
  })
  .strict();
export type ImagePart = z.infer<typeof ImagePartSchema>;

export const ToolCallPartSchema = z
  .object({
    type: z.literal("tool_call"),
    id: z.string().min(1).max(128),
    name: z.string().min(1).max(64),
    args: z.record(z.unknown()),
    status: ToolCallStatusSchema,
  })
  .strict();
export type ToolCallPart = z.infer<typeof ToolCallPartSchema>;

export const ToolResultPartSchema = z
  .object({
    type: z.literal("tool_result"),
    call_id: z.string().min(1).max(128),
    summary: z.string().max(1000),
    data: z.record(z.unknown()),
  })
  .strict();
export type ToolResultPart = z.infer<typeof ToolResultPartSchema>;

export const ContentPartSchema = z.discriminatedUnion("type", [
  TextPartSchema,
  ImagePartSchema,
  ToolCallPartSchema,
  ToolResultPartSchema,
]);
export type ContentPart = z.infer<typeof ContentPartSchema>;

export const UserContentPartSchema = z.discriminatedUnion("type", [
  TextPartSchema,
  ImagePartSchema,
]);
export type UserContentPart = z.infer<typeof UserContentPartSchema>;

export const MessageSchema = z
  .object({
    id: z.string().min(1).max(128),
    conversation_id: z.string().min(1).max(128),
    role: RoleSchema,
    parts: z.array(ContentPartSchema).min(1).max(32),
    created_at: z.string(),
    parent_id: z.string().max(128).nullable().optional(),
  })
  .strict();
export type Message = z.infer<typeof MessageSchema>;
