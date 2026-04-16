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
