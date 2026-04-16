import {
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
