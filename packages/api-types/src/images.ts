import { z } from "zod";

/** `POST /images` のレスポンス。返ってきた `image_id` を ImagePart に詰めて chat に送る。 */
export const ImageUploadResponseSchema = z
  .object({
    image_id: z.string(),
    image_url: z.string(),
    mime: z.string(),
    width: z.number().int(),
    height: z.number().int(),
  })
  .strict();

export type ImageUploadResponse = z.infer<typeof ImageUploadResponseSchema>;

/** ギャラリー一覧で返る 1 件分の生成画像メタ。`created_at` は ISO8601 文字列。 */
export const GeneratedImageRefSchema = z
  .object({
    image_id: z.string(),
    image_url: z.string(),
    conversation_id: z.string(),
    created_at: z.string(),
    prompt_excerpt: z.string(),
    seed: z.number().int(),
    aspect_ratio: z.enum(["portrait", "landscape", "square"]),
    cfg_preset: z.enum(["soft", "standard", "crisp"]),
    width: z.number().int(),
    height: z.number().int(),
  })
  .strict();

export type GeneratedImageRef = z.infer<typeof GeneratedImageRefSchema>;

/** `GET /images` のレスポンス。`next_before` を `?before=` に渡すと続きが取れる。 */
export const GeneratedImageListSchema = z
  .object({
    images: z.array(GeneratedImageRefSchema),
    next_before: z.string().nullable(),
  })
  .strict();

export type GeneratedImageList = z.infer<typeof GeneratedImageListSchema>;
