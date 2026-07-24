import * as z from 'zod';
import { ContextParamsSchema, ModelParamsSchema, SamplerConstructorScheme } from './llama-base.js';
import { InferenceLineParamsScheme, StopReasonsSchema } from './model.js';


export const SToken = z.object({
    token: z.int().nonnegative(),
    piece: z.string(),
    control: z.boolean(),
});
export const SEventArgs = {
    command_json_error: z.object({ message: z.string() }),
    command_schema_error: z.object({ message: z.string(), issues: z.array(z.unknown()) }),
    preload_done: z.object({ time_ms: z.int().nonnegative() }),
    model_loaded: z.object({ time_ms: z.union([z.int().nonnegative(), z.null()]), model_file: z.string() }),
    model_loading: z.object({ progress: z.number().min(0).max(1) }),
    llama_log: z.object({ message: z.string(), level: z.enum(["none", "debug", "info", "warn", "error"]) }),
    ggml_log: z.object({ message: z.string(), level: z.enum(["none", "debug", "info", "warn", "error"]) }),
    ctx_changed: z.object({ context_params: ContextParamsSchema }),
    tokens: z.object({
        line_id: z.string(),
        next: z.union([z.null(), SToken]),
        input: z.array(SToken),
        entropy: z.union([z.null(), z.number().nonnegative()]),
        stop: z.boolean(),
        stopReasons: z.array(StopReasonsSchema),
    }),
    generation_started: z.null(),
    generation_stopped: z.null(),
};
export const SEventSchema = z.discriminatedUnion("event", [
    z.object({ event: z.literal("command_json_error"), args: SEventArgs["command_json_error"] }),
    z.object({ event: z.literal("command_schema_error"), args: SEventArgs["command_schema_error"] }),
    z.object({ event: z.literal("preload_done"), args: SEventArgs["preload_done"] }),
    z.object({ event: z.literal("model_loaded"), args: SEventArgs["model_loaded"] }),
    z.object({ event: z.literal("model_loading"), args: SEventArgs["model_loading"] }),
    z.object({ event: z.literal("llama_log"), args: SEventArgs["llama_log"] }),
    z.object({ event: z.literal("ggml_log"), args: SEventArgs["ggml_log"] }),
    z.object({ event: z.literal("ctx_changed"), args: SEventArgs["ctx_changed"] }),
    z.object({ event: z.literal("tokens"), args: SEventArgs["tokens"] }),
    z.object({ event: z.literal("generation_started"), args: SEventArgs["generation_started"] }),
    z.object({ event: z.literal("generation_stopped"), args: SEventArgs["generation_stopped"] }),
]);
export type SEvent = z.infer<typeof SEventSchema>;




export const SResultArgs = {
    start: z.object({
        model_params: ModelParamsSchema,
        metadata: z.record(z.string(), z.string()),
    }),
    set_context: z.null(),
    line_init: z.object({
        line_id: z.string(),
    }),
    line_free: z.null(),
    line_list: z.array(z.object({
        line_id: z.string(),
    })),
    line_load: z.null(),
    line_save: z.object({
        path: z.string(),
    }),
    line_push: z.null(),
    line_trim: z.null(),
    line_cancel: z.null(),
    line_start: z.null(),
    line_stop: z.null(),
    line_clear: z.null(),
    tokenize: z.object({
        tokens: z.array(z.number().nonnegative()),
    }),
    detokenize: z.object({
        text: z.string(),
    }),
    exit: z.null(),
};
export const SResultSchema = z.discriminatedUnion("command", [
    z.object({ command: z.literal("start"), args: SResultArgs["start"] }),
    z.object({ command: z.literal("set_context"), args: SResultArgs["set_context"] }),
    z.object({ command: z.literal("line_init"), args: SResultArgs["line_init"] }),
    z.object({ command: z.literal("line_list"), args: SResultArgs["line_list"] }),
    z.object({ command: z.literal("line_free"), args: SResultArgs["line_free"] }),
    z.object({ command: z.literal("line_load"), args: SResultArgs["line_load"] }),
    z.object({ command: z.literal("line_save"), args: SResultArgs["line_save"] }),
    z.object({ command: z.literal("line_push"), args: SResultArgs["line_push"] }),
    z.object({ command: z.literal("line_trim"), args: SResultArgs["line_trim"] }),
    z.object({ command: z.literal("line_cancel"), args: SResultArgs["line_cancel"] }),
    z.object({ command: z.literal("line_start"), args: SResultArgs["line_start"] }),
    z.object({ command: z.literal("line_stop"), args: SResultArgs["line_stop"] }),
    z.object({ command: z.literal("line_clear"), args: SResultArgs["line_clear"] }),
    z.object({ command: z.literal("tokenize"), args: SResultArgs["tokenize"] }),
    z.object({ command: z.literal("detokenize"), args: SResultArgs["detokenize"] }),
    z.object({ command: z.literal("exit"), args: SResultArgs["exit"] }),
]);
export type SResult = z.infer<typeof SResultSchema>;



export const SErrorArgsFields = {
    internal_error: z.record(z.string(), z.unknown()),
    too_many_lines: z.object({
        max_lines: z.int().positive(),
    }),
    line_not_found: z.object({
        line_id: z.string(),
    })
};
export const SErrorArgs = {
    internal_error: z.object({ message: z.string(), fields: SErrorArgsFields["internal_error"] }),
    too_many_lines: z.object({ message: z.string(), fields: SErrorArgsFields["too_many_lines"] }),
    line_not_found: z.object({ message: z.string(), fields: SErrorArgsFields["line_not_found"] }),
};
export const SErrorSchema = z.discriminatedUnion("error", [
    z.object({ error: z.literal("internal_error"), args: SErrorArgs["internal_error"] }),
    z.object({ error: z.literal("too_many_lines"), args: SErrorArgs["too_many_lines"] }),
    z.object({ error: z.literal("line_not_found"), args: SErrorArgs["line_not_found"] }),
]);
export type SError = z.infer<typeof SErrorSchema>;




export const SCommandArgs = {
    start: z.null(),
    set_context: z.object({ context_params: ContextParamsSchema }),
    line_init: z.object({
        line_id: z.string().optional(),
        sampler: SamplerConstructorScheme.optional(),
        sampler_offset: z.int().nonnegative().optional(),
        inference: InferenceLineParamsScheme.optional(),
    }),
    line_free: z.object({
        line_id: z.string(),
    }),
    line_list: z.null(),
    line_load: z.object({
        line_id: z.string(),
        path: z.string(),
    }),
    line_save: z.object({
        line_id: z.string(),
        path: z.string().optional(),
    }),
    line_push: z.object({
        line_id: z.string(),
        tokens: z.array(z.int().nonnegative()),
    }),
    line_trim: z.object({
        line_id: z.string(),
        n_tokens: z.int().nonnegative(),
    }),
    line_cancel: z.object({
        line_id: z.string(),
    }),
    line_start: z.object({
        line_id: z.string(),
    }),
    line_stop: z.object({
        line_id: z.string(),
    }),
    line_clear: z.object({
        line_id: z.string(),
    }),
    tokenize: z.object({
        text: z.string(),
        parse_special: z.boolean().optional(),
        add_special: z.boolean().optional(),
    }),
    detokenize: z.object({
        tokens: z.array(z.number()),
        unparse_special: z.boolean().optional(),
        remove_special: z.boolean().optional(),
    }),
    exit: z.null(),
};
export const SCommandSchema = z.discriminatedUnion("command", [
    z.object({ command: z.literal("start"), query_id: z.string(), args: SCommandArgs["start"] }),
    z.object({ command: z.literal("set_context"), query_id: z.string(), args: SCommandArgs["set_context"] }),
    z.object({ command: z.literal("line_init"), query_id: z.string(), args: SCommandArgs["line_init"] }),
    z.object({ command: z.literal("line_free"), query_id: z.string(), args: SCommandArgs["line_free"] }),
    z.object({ command: z.literal("line_list"), query_id: z.string(), args: SCommandArgs["line_list"] }),
    z.object({ command: z.literal("line_load"), query_id: z.string(), args: SCommandArgs["line_load"] }),
    z.object({ command: z.literal("line_save"), query_id: z.string(), args: SCommandArgs["line_save"] }),
    z.object({ command: z.literal("line_push"), query_id: z.string(), args: SCommandArgs["line_push"] }),
    z.object({ command: z.literal("line_trim"), query_id: z.string(), args: SCommandArgs["line_trim"] }),
    z.object({ command: z.literal("line_cancel"), query_id: z.string(), args: SCommandArgs["line_cancel"] }),
    z.object({ command: z.literal("line_start"), query_id: z.string(), args: SCommandArgs["line_start"] }),
    z.object({ command: z.literal("line_stop"), query_id: z.string(), args: SCommandArgs["line_stop"] }),
    z.object({ command: z.literal("line_clear"), query_id: z.string(), args: SCommandArgs["line_clear"] }),
    z.object({ command: z.literal("tokenize"), query_id: z.string(), args: SCommandArgs["tokenize"] }),
    z.object({ command: z.literal("detokenize"), query_id: z.string(), args: SCommandArgs["detokenize"] }),
    z.object({ command: z.literal("exit"), query_id: z.string(), args: SCommandArgs["exit"] }),
]);
export type SCommand = z.infer<typeof SCommandSchema>;

















export const SMessageSchema = z.discriminatedUnion("type", [
    z.object({ type: z.literal("event"), object: SEventSchema }),
    z.object({ type: z.literal("result"), object: SResultSchema, query_id: z.string() }),
    z.object({ type: z.literal("error"), object: SErrorSchema, query_id: z.string() }),
]);
export type SMessage = z.infer<typeof SMessageSchema>;


//