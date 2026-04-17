import {
  type ConversationDetail,
  ConversationDetailSchema,
  type GeneratedImageList,
  GeneratedImageListSchema,
  type ImageUploadResponse,
  ImageUploadResponseSchema,
} from "@cocktail/api-types";

/** `POST /images` で 1 枚アップロードし、`image_id` を受け取る。 */
export async function uploadImage(file: File): Promise<ImageUploadResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch("/images", { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Upload failed (${res.status}): ${body}`);
  }
  const json: unknown = await res.json();
  return ImageUploadResponseSchema.parse(json);
}

/** `GET /images` で生成済み画像の横断一覧を取得する（`created_at` 降順）。 */
export async function listGeneratedImages(
  limit = 50,
  before?: string,
): Promise<GeneratedImageList> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (before) params.set("before", before);
  const res = await fetch(`/images?${params.toString()}`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`List failed (${res.status}): ${body}`);
  }
  const json: unknown = await res.json();
  return GeneratedImageListSchema.parse(json);
}

/** `GET /conversations/{id}` で会話 1 件の詳細（messages + 生成画像メタ）を取得する。 */
export async function getConversation(id: string): Promise<ConversationDetail> {
  const res = await fetch(`/conversations/${encodeURIComponent(id)}`, {
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Get conversation failed (${res.status}): ${body}`);
  }
  const json: unknown = await res.json();
  return ConversationDetailSchema.parse(json);
}
