import { z } from "zod";

/**
 * `apps/server/src/cocktail_server/schemas/health.py` に対応する Zod スキーマ。
 * サーバ側を変更したときはここも合わせて更新すること（openapi.json の
 * `HealthResponse` と形が一致していれば OK）。
 */

export const StartupStateSchema = z.enum(["downloading", "loading", "ready", "error"]);
export type StartupState = z.infer<typeof StartupStateSchema>;

export const ModelStatusSchema = z.enum(["loaded", "loading", "idle", "error"]);
export type ModelStatus = z.infer<typeof ModelStatusSchema>;

export const ResidencyPolicySchema = z.enum(["swap", "coresident"]);
export type ResidencyPolicy = z.infer<typeof ResidencyPolicySchema>;

export const StartupStatusSchema = z.object({
  state: StartupStateSchema,
  error: z.string().nullable(),
});
export type StartupStatus = z.infer<typeof StartupStatusSchema>;

export const GpuInfoSchema = z.object({
  name: z.string(),
  memory_used_mb: z.number().int(),
  memory_total_mb: z.number().int(),
  vram_total_gb: z.number().nullable(),
  vram_free_gb: z.number().nullable(),
});
export type GpuInfo = z.infer<typeof GpuInfoSchema>;

export const ModelsStatusSchema = z.object({
  llm: ModelStatusSchema,
  image: ModelStatusSchema,
});
export type ModelsStatus = z.infer<typeof ModelsStatusSchema>;

export const HealthResponseSchema = z.object({
  startup: StartupStatusSchema,
  gpu: GpuInfoSchema.nullable(),
  models: ModelsStatusSchema,
  queue_depth: z.number().int(),
  residency_policy: ResidencyPolicySchema,
});
export type HealthResponse = z.infer<typeof HealthResponseSchema>;
