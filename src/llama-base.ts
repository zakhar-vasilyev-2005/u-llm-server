import EventEmitter from "events";
import { address, alloc, array, decode, encode, free, load, out, pointer, proto, register, sizeof, struct, unregister, view, type LibraryHandle } from "koffi";
import fs from 'fs-extra';
import { createFreeEvent } from "./event-util.js";
import { Yurandom } from 'yurandom';
import * as z from "zod";
import path from "path";

const Ptr = "void*", Void = "void", Str = "char*", Char = "char";
const Int = "int32_t", Long = "int64_t", Uint = "uint32_t", Ulong = "uint64_t", Size = "size_t", Byte = "uint8_t";
const Float = "float", Double = "double", Bool = "bool";





export class Lib extends EventEmitter<{ lib_free: [] }> {
    public readonly lib: LibraryHandle;
    public constructor(public readonly file: string) {
        if (!fs.existsSync(file)) {
            const dir = path.dirname(file);
            const base = path.basename(file);
            let disableError = false;
            if (fs.existsSync(dir)) {
                const m = fs.readdirSync(dir).map(e => e.startsWith(base) ? /^(\.\d+)+$/.exec(e.slice(base.length)) : null).find(e => e !== null);
                if (m !== undefined) {
                    file = file + m[0];
                    disableError = true;
                }
            }
            if (!disableError) {
                throw new Error(`no such file found: ${JSON.stringify(file)}`);
            }
        }
        super();
        this.lib = load(file);
        this.file = file;
    }
    public free = createFreeEvent<this, void>("lib_free", ths => ths.lib.unload());
}

export const GGMLLogLevels = { none: 0, debug: 1, info: 2, warn: 3, error: 4, cont: 5 };
export type GGMLLogLevel = keyof typeof GGMLLogLevels;

export class LibGGML extends Lib {
    public log_set(logCallback: (level: GGMLLogLevel, message: string) => void) {
        const lvTable = (Object.entries(GGMLLogLevels) as [GGMLLogLevel, number][]);
        const cbType = pointer(proto(Void, [Int, Str]));
        const cb = register((lv: number, msg: string) => {
            logCallback(lvTable.find(e => e[1] === lv)?.[0] ?? "none", msg);
        }, cbType);
        this.lib.func("ggml_log_set", Void, [cbType, Ptr])(cb, null);
    }
    public backend_load_all_from_path(p: string) {
        if (!fs.existsSync(p)) { throw new Error(`requested path (${JSON.stringify(p)}) not exists`); }
        this.lib.func("ggml_backend_load_all_from_path", Void, [Str])(p);
    }
}
export class LibLlama extends Lib {
    public log_set(logCallback: (level: GGMLLogLevel, message: string) => void): void {
        const lvTable = (Object.entries(GGMLLogLevels) as [GGMLLogLevel, number][]);
        const cbType = pointer(proto(Void, [Int, Str]));
        const cb = register((lv: number, msg: string) => {
            logCallback(lvTable.find(e => e[1] === lv)?.[0] ?? "none", msg);
        }, cbType);
        this.lib.func("llama_log_set", Void, [cbType, Ptr])(cb, null);
    }
    public backend_init(): void {
        this.lib.func("llama_backend_init", Void, [])();
    }
    public backend_free(): void {
        this.lib.func("llama_backend_free", Void, [])();
    }
    public model_default_params(): any {
        return this.lib.func("llama_model_default_params", ModelParamsStruct, [])()
    }
    public context_default_params(): any {
        return this.lib.func("llama_context_default_params", ContextParamsStruct, [])();
    }
    public context_params(cparams: ContextParams): any {
        return Object.fromEntries(Object.entries(cparams).map(([key, value]) => {
            if (key === "type_k" || key === "type_v") {
                const index = (GGMLTypes as any)[value as string] as number | undefined;
                if (index === undefined) { throw new Error(`unexpected situation`); }
                return [key, index];
            } else if (key === "ctx_type") {
                return [key, ["default", "mtp"].indexOf(value as string)];
            } else if (key === "rope_scaling_type") {
                if (value === "max_value") { value = "longrope"; }
                return [key, ["none", "linear", "yarn", "longrope"].indexOf(value as string)];
            } else if (key === "pooling_typ") {
                return [key, ["none", "mean", "cls", "last", "rank"].indexOf(value as string)];
            } else if (key === "attention_type") {
                return [key, ["casual", "non_casual"].indexOf(value as string)];
            } else if (key === "flash_attn_type") {
                return [key, ["disabled", "enabled"].indexOf(value as string)];
            } else {
                return [key, value];
            }
        }));
    }
    public init_from_model(model_ptr: bigint, context_params: any): bigint | null {
        return this.lib.func("llama_init_from_model", Ptr, [Ptr, ContextParamsStruct])(model_ptr, context_params);
    }
    public n_seq_max(context_ptr: bigint): number {
        return this.lib.func("llama_n_seq_max", Uint, [Ptr])(context_ptr);
    }
    public get_memory(context_ptr: bigint): bigint {
        return this.lib.func("llama_get_memory", Ptr, [Ptr])(context_ptr);
    }
    public memory_seq_pos_min(memory_ptr: bigint, line_id: number): number {
        return this.lib.func("llama_memory_seq_pos_min", Int, [Ptr, Int])(memory_ptr, line_id);
    }
    public memory_seq_pos_max(memory_ptr: bigint, line_id: number): number {
        return this.lib.func("llama_memory_seq_pos_max", Int, [Ptr, Int])(memory_ptr, line_id);
    }
    public memory_seq_rm(memory_ptr: bigint, line_id: number, p1: number, p2: number): boolean {
        return this.lib.func("llama_memory_seq_rm", Bool, [Ptr, Int, Int, Int])(memory_ptr, line_id, p1, p2);
    }
    public state_seq_get(context_ptr: bigint, line_id: number): Buffer {
        const buffer = Buffer.alloc(this.lib.func("llama_state_seq_get_size", Size, [Ptr, Int])(context_ptr, line_id));
        this.lib.func("llama_state_seq_get_data", Size, [Ptr, pointer(Byte), Size, Int])(context_ptr, buffer, buffer.byteLength, line_id);
        return buffer;
    }
    public state_seq_set(context_ptr: bigint, line_id: number, state: Buffer): void {
        this.memory_seq_rm(this.get_memory(context_ptr), line_id, 0, -1);
        this.lib.func("llama_state_seq_set_data", Size, [Ptr, pointer(Byte), Size, Int])(context_ptr, state, state.byteLength, line_id);
    }
    public state_seq_save_file(context_ptr: bigint, line_id: number, tokens: number[], file: string): void {
        this.lib.func("llama_state_seq_save_file", Size, [Ptr, Str, Int, pointer(Int), Size])(context_ptr, file, line_id, tokens, tokens.length);
    }
    public static state_seq_load_file_buffer = new Int32Array(1024 * 1024);
    public state_seq_load_file(context_ptr: bigint, line_id: number, file: string): Int32Array {
        const size_out = [0];
        this.memory_seq_rm(this.get_memory(context_ptr), line_id, 0, -1);
        this.lib.func("llama_state_seq_load_file", Size, [Ptr, Str, Int, out(pointer(Int)), Size, out(pointer(Size))])(
            context_ptr, file, line_id, LibLlama.state_seq_load_file_buffer, LibLlama.state_seq_load_file_buffer.length, size_out
        );
        return LibLlama.state_seq_load_file_buffer.slice(0, size_out[0]);
    }
    public state_seq_trim(context_ptr: bigint, line_id: number, n_tokens: number): void {
        const memory_ptr = this.get_memory(context_ptr);
        const start_pos = this.memory_seq_pos_max(memory_ptr, line_id) + 1 - n_tokens;
        this.memory_seq_rm(memory_ptr, line_id, start_pos, -1);
    }
    public context_free(context_ptr: bigint): void {
        this.lib.func("llama_free", Void, [Ptr])(context_ptr);
    }
    public model_load_from_file(file: string, params: any): bigint | null {
        const fn = this.lib.func("llama_model_load_from_file", Ptr, [Str, ModelParamsStruct]);
        params = Object.assign({}, params);
        const cb = params["progress_callback"];
        if (typeof cb === "function") {
            const native = register(cb, pointer(proto(Bool, [Float, Ptr])));
            params["progress_callback"] = native;
            const ptr = fn(file, params);
            unregister(native);
            return ptr;
        } else {
            return fn(file, params);
        }
    }
    public model_free(ptr: bigint): void {
        this.lib.func("llama_model_free", Void, [Ptr])(ptr);
    }
    public model_meta_count(model_ptr: bigint): number {
        return this.lib.func("llama_model_meta_count", Int, [Ptr])(model_ptr);
    }
    public model_meta_key_by_index(model_ptr: bigint, idx: number): string {
        const fn = this.lib.func("llama_model_meta_key_by_index", Int, [Ptr, Int, Str, Size]);
        const size = fn(model_ptr, idx, null, 0);
        const buffer = Buffer.alloc(size + 1);
        fn(model_ptr, idx, buffer, buffer.byteLength);
        return buffer.toString("utf8").slice(0, -1);
    }
    public model_meta_val_str_by_index(model_ptr: bigint, idx: number): string {
        const fn = this.lib.func("llama_model_meta_val_str_by_index", Int, [Ptr, Int, Str, Size]);
        const size = fn(model_ptr, idx, null, 0);
        const buffer = Buffer.alloc(size + 1);
        fn(model_ptr, idx, buffer, buffer.byteLength);
        return buffer.toString("utf8").slice(0, -1);
    }
    public model_n_ctx_train(model_ptr: bigint): number {
        return this.lib.func("llama_model_n_ctx_train", Int, [Ptr])(model_ptr);
    }
    public model_get_vocab(model_ptr: bigint): bigint {
        return this.lib.func("llama_model_get_vocab", Ptr, [Ptr])(model_ptr) as bigint;
    }
    public vocab_is_eog(vocab_ptr: bigint, token: number): boolean {
        return this.lib.func("llama_vocab_is_eog", Bool, [Ptr, Int])(vocab_ptr, token) as boolean;
    }
    public vocab_is_control(vocab_ptr: bigint, token: number): boolean {
        return this.lib.func("llama_vocab_is_control", Bool, [Ptr, Int])(vocab_ptr, token) as boolean;
    }
    public vocab_n_tokens(vocab_ptr: bigint): number {
        return this.lib.func("llama_vocab_n_tokens", Int, [Ptr])(vocab_ptr) as number;
    }
    public detokenize(vocab_ptr: bigint, tokens: number[], unparse_special: boolean, remove_special: boolean = false): Buffer {
        const fn = (buffer: Buffer) => this.lib.func("llama_detokenize", Int, [Ptr, pointer(Int), Int, out(pointer(Char)), Int, Bool, Bool])(
            vocab_ptr, tokens, tokens.length, buffer, buffer.byteLength, remove_special, unparse_special
        );
        if (tokens.length === 0) { return Buffer.alloc(0); }
        const size = - fn(Buffer.alloc(0));
        const buffer = Buffer.alloc(size + 1);
        fn(buffer);
        return buffer.subarray(0, -1)
    }
    public tokenize(vocab_ptr: bigint, text: string, parse_special: boolean, add_special: boolean = false): Int32Array {
        const textBuffer = Buffer.from(text, "utf8");
        const fn = (buffer: Int32Array) => this.lib.func("llama_tokenize", Int, [Ptr, Str, Int, out(pointer(Int)), Int, Bool, Bool])(
            vocab_ptr, textBuffer, textBuffer.byteLength, buffer, buffer.length, add_special, parse_special
        );
        const n_tokens = - fn(new Int32Array(0));
        const buffer = new Int32Array(n_tokens);
        fn(buffer);
        return buffer;
    }
    public static token_to_piece_buffer = Buffer.alloc(512);
    public token_to_piece(vocab_ptr: bigint, token: number, renderSpecial: boolean = true): string {
        return this.token_to_piece_raw(vocab_ptr, token, renderSpecial).toString();
    }
    public token_to_piece_raw(vocab_ptr: bigint, token: number, renderSpecial: boolean = true): Buffer {
        const len = this.lib.func("llama_token_to_piece", Int, [Ptr, Int, out(Str), Int, Int, Bool])(
            vocab_ptr, token, LibLlama.token_to_piece_buffer, 512, 0, renderSpecial
        );
        return LibLlama.token_to_piece_buffer.subarray(0, len);
    }
    public decode(context_ptr: bigint, batch: any): number {
        return this.lib.func("llama_decode", Int, [Ptr, BatchStruct])(context_ptr, batch);
    }
    public get_logits_ith(context_ptr: bigint, vocab_size: number, i: number): Float32Array {
        const ptr = this.lib.func("llama_get_logits_ith", Ptr, [Ptr, Int])(context_ptr, i);
        return new Float32Array(view(ptr, 4 * vocab_size));
    }
    public sampler_chain_init(): bigint {
        return this.lib.func("llama_sampler_chain_init", Ptr, [struct({ no_perf: Bool })])({ no_perf: true });
    }
    public sampler_chain_add(chain: bigint, elem: bigint): void {
        this.lib.func("llama_sampler_chain_add", Void, [Ptr, Ptr])(chain, elem);
    }
    public sampler_init_greedy(): bigint {
        return this.lib.func("llama_sampler_init_greedy", Ptr, [])();
    }
    public sampler_init_dist(seed: number): bigint {
        return this.lib.func("llama_sampler_init_dist", Ptr, [Uint])(seed);
    }
    public sampler_init_top_k(k: number): bigint {
        return this.lib.func("llama_sampler_init_top_k", Ptr, [Int])(k);
    }
    public sampler_init_top_p(p: number, min_keep: number): bigint {
        return this.lib.func("llama_sampler_init_top_p", Ptr, [Float, Size])(p, min_keep);
    }
    public sampler_init_min_p(p: number, min_keep: number): bigint {
        return this.lib.func("llama_sampler_init_min_p", Ptr, [Float, Size])(p, min_keep);
    }
    public sampler_init_typical(p: number, min_keep: number): bigint {
        return this.lib.func("llama_sampler_init_typical", Ptr, [Float, Size])(p, min_keep);
    }
    public sampler_init_temp(t: number): bigint {
        return this.lib.func("llama_sampler_init_temp", Ptr, [Float])(t);
    }
    public sampler_init_temp_ext(t: number, delta: number, exponent: number): bigint {
        return this.lib.func("llama_sampler_init_temp_ext", Ptr, [Float, Float, Float])(t, delta, exponent);
    }
    public sampler_init_xtc(p: number, t: number, min_keep: number, seed: number): bigint {
        return this.lib.func("llama_sampler_init_xtc", Ptr, [Float, Float, Size, Uint])(p, t, min_keep, seed);
    }
    public sampler_init_top_n_sigma(n: number): bigint {
        return this.lib.func("llama_sampler_init_top_n_sigma", Ptr, [Float])(n);
    }
    public sampler_init_mirostat(n_vocab: number, seed: number, tau: number, eta: number, m: number): bigint {
        return this.lib.func("llama_sampler_init_mirostat", Ptr, [Int, Uint, Float, Float, Int])(n_vocab, seed, tau, eta, m);
    }
    public sampler_init_mirostat_v2(seed: number, tau: number, eta: number): bigint {
        return this.lib.func("llama_sampler_init_mirostat_v2", Ptr, [Uint, Float, Float])(seed, tau, eta);
    }
    public sampler_init_grammar(vocab_ptr: bigint, grammar_str: string, grammar_root: string): bigint {
        return this.lib.func("llama_sampler_init_grammar", Ptr, [Ptr, Str, Str])(vocab_ptr, grammar_str, grammar_root);
    }
    public sampler_init_grammar_lazy_patterns(vocab_ptr: bigint, grammar_str: string, grammar_root: string, triggers: (string | number)[]): bigint {
        const token_triggers = triggers.filter(e => typeof e === "number");
        const str_triggers = triggers.filter(e => typeof e === "string");
        return this.lib.func("llama_sampler_init_grammar_lazy_patterns", Ptr, [Ptr, Str, Str, pointer(Str), Size, pointer(Int), Size])(
            vocab_ptr, grammar_str, grammar_root, str_triggers, str_triggers.length, token_triggers, token_triggers.length
        );
    }
    public sampler_init_penalties(last_n: number, repeat: number = 1, freq: number = 0, present: number = 0): bigint {
        return this.lib.func("llama_sampler_init_penalties", Ptr, [Uint, Float, Float, Float])(last_n, repeat, freq, present);
    }
    public sampler_init_dry(vocab_ptr: bigint, n_ctx_train: number, dry_multiplier: number, dry_base: number, dry_allowed_length: number, penalty_last_n: number, seq_breakers: string[]): bigint {
        return this.lib.func("llama_sampler_init_dry", Ptr, [Ptr, Int, Float, Float, Int, Int, pointer(Str), Size])(
            vocab_ptr, n_ctx_train, dry_multiplier, dry_base, dry_allowed_length, penalty_last_n, seq_breakers, seq_breakers.length
        );
    }
    public sampler_init_adaptive_p(target: number, decay: number, seed: number): bigint {
        return this.lib.func("llama_sampler_init_adaptive_p", Ptr, [Float, Float, Uint])(target, decay, seed);
    }
    public sampler_init_logit_bias(n_vocab: number, logit_bias: { token: number, bias: number }[]): bigint {
        return this.lib.func("llama_sampler_init_penalties", Ptr, [Int, Int, pointer(struct({ token: Int, bias: Float }))])(n_vocab, logit_bias.length, logit_bias);
    }
    public sampler_init_infill(vocab_ptr: bigint): bigint {
        return this.lib.func("llama_sampler_init_penalties", Ptr, [Ptr])(vocab_ptr);
    }
    public sampler_chain(model_ptr: bigint, sampler: SamplerConstructor, sampler_seed: number | string = `${process.pid}_${Date.now()}`): bigint {
        const chain = this.sampler_chain_init();
        const vocab_ptr = this.model_get_vocab(model_ptr);
        const n_vocab = this.vocab_n_tokens(vocab_ptr);
        const seed = Math.abs(typeof sampler_seed === "string" ? new Yurandom(sampler_seed).int(0, 32000) : sampler_seed) % 32000;
        for (const elem of sampler) {
            let e: bigint | null = null;
            if (elem.type === "greedy") {
                e = this.sampler_init_greedy();
            } else if (elem.type === "dist") {
                e = this.sampler_init_dist(elem.seed ?? seed);
            } else if (elem.type === "top_k") {
                e = this.sampler_init_top_k(elem.k);
            } else if (elem.type === "top_p") {
                e = this.sampler_init_top_p(elem.p, elem.min_keep);
            } else if (elem.type === "min_p") {
                e = this.sampler_init_min_p(elem.p, elem.min_keep);
            } else if (elem.type === "typical") {
                e = this.sampler_init_typical(elem.p, elem.min_keep);
            } else if (elem.type === "temp") {
                e = this.sampler_init_temp(elem.t);
            } else if (elem.type === "temp_ext") {
                e = this.sampler_init_temp_ext(elem.t, elem.delta, elem.exponent);
            } else if (elem.type === "xtc") {
                e = this.sampler_init_xtc(elem.p, elem.t, elem.min_keep, elem.seed ?? seed);
            } else if (elem.type === "top_n_sigma") {
                e = this.sampler_init_top_n_sigma(elem.n);
            } else if (elem.type === "mirostat") {
                e = this.sampler_init_mirostat(n_vocab, elem.seed ?? seed, elem.tau, elem.eta, elem.m);
            } else if (elem.type === "mirostat_v2") {
                e = this.sampler_init_mirostat_v2(elem.seed, elem.tau, elem.eta);
            } else if (elem.type === "grammar") {
                e = this.sampler_init_grammar(vocab_ptr, elem.grammar, elem.root ?? "root");
            } else if (elem.type === "grammar_lazy_patterns") {
                e = this.sampler_init_grammar_lazy_patterns(vocab_ptr, elem.grammar, elem.root ?? "root", elem.triggers);
            } else if (elem.type === "penalties") {
                e = this.sampler_init_penalties(elem.last_n, elem.repeat ?? 1, elem.freq ?? 0, elem.present ?? 0);
            } else if (elem.type === "dry") {
                e = this.sampler_init_dry(vocab_ptr, this.model_n_ctx_train(model_ptr), elem.multiplier, elem.base, elem.allowed_length, elem.penalty_last_n, elem.seq_breakers);
            } else if (elem.type === "adaptive_p") {
                e = this.sampler_init_adaptive_p(elem.target, elem.decay, elem.seed ?? seed);
            } else if (elem.type === "logit_bias") {
                e = this.sampler_init_logit_bias(n_vocab, elem.logit_bias);
            } else if (elem.type === "infill") {
                e = this.sampler_init_infill(vocab_ptr);
            } else {
                throw new Error(`bad sampler chain elem: ${JSON.stringify(elem)}`);
            }
            this.sampler_chain_add(chain, e);
        }
        return chain;
    }
    public sampler_free(sampler_ptr: bigint): void {
        this.lib.func("llama_sampler_free", Void, [Ptr])(sampler_ptr);
    }
    public sampler_apply(sampler_ptr: bigint, cur_pt: bigint): void {
        this.lib.func("llama_sampler_apply", Void, [Ptr, Ptr])(sampler_ptr, cur_pt);
    }
    public sampler_accept(sampler_ptr: bigint, token: number): void {
        this.lib.func("llama_sampler_accept", Void, [Ptr, Int])(sampler_ptr, token);
    }
}
export class LibEntropyRaw extends Lib {
    public entropyOfLogits(logits: Float32Array) {
        return this.lib.func("entropy_of_logits", Double, [pointer(Float), Size])(logits, logits.length) as number;
    }
}
export class LibEntropy {
    public raw: LibEntropyRaw | undefined;
    public constructor(libraryFile: string | undefined) {
        this.raw = libraryFile === undefined ? undefined : new LibEntropyRaw(libraryFile);
    }
    public entropyOfLogits(logits: Float32Array) {
        if (this.raw !== undefined) {
            return this.raw.entropyOfLogits(logits);
        } else {
            const divisor = logits.map(e => Math.exp(e)).reduce((a, b) => a + b, 0);
            const p = logits.map(e => Math.exp(e) / divisor);
            const entropy = - p.map(e => e * Math.log(e)).reduce((a, b) => a + b, 0);
            return entropy;
        }
    }
}
export class LibSamplingHelperRaw extends Lib {
    public logitsToCurp(logits: Float32Array): bigint {
        return this.lib.func("logits_to_curp", Ptr, [pointer(Float), Size])(logits, logits.length);
    }
    public curpToToken(cur_p: bigint): number {
        return this.lib.func("curp_to_token", Int, [Ptr])(cur_p);
    }
}
export class LibSamplingHelper {
    public raw: LibSamplingHelperRaw | undefined;
    public constructor(libraryFile: string | undefined) {
        this.raw = libraryFile === undefined ? undefined : new LibSamplingHelperRaw(libraryFile);
    }
    public allocated: { ptr: bigint, handle: any }[] = [];
    public logitsToCurp(logits: Float32Array): bigint {
        if (this.raw !== undefined) {
            return this.raw.logitsToCurp(logits);
        } else {
            const data = alloc(TokenData, logits.length);
            this.allocated.push({ ptr: address(data), handle: data });
            encode(data, array(TokenData, logits.length), [...logits].map((logit, id) => ({ id, logit, p: 0 })));
            const cur_p = alloc(TokenDataArray, 1);
            this.allocated.push({ ptr: address(cur_p), handle: cur_p });
            encode(cur_p, TokenDataArray, { data, size: logits.length, selected: -1, sorted: false, });
            return address(cur_p);
        }
    }
    public curpToToken(cur_p: bigint): number {
        if (this.raw !== undefined) {
            return this.raw.curpToToken(cur_p);
        } else {
            const { selected, data } = decode(cur_p, TokenDataArray) as { selected: number, data: bigint };
            const entry = decode(data, sizeof(TokenData) * selected, TokenData);
            free(data);
            free(cur_p);
            this.allocated = this.allocated.filter(e => e.ptr !== data && e.ptr !== cur_p);
            return entry.id as number;
        }
    }
}

export function createGGMLLogger(levels: GGMLLogLevel[], log: (message: string, level: "none" | "debug" | "info" | "warn" | "error") => void) {
    let prevLevel: "none" | "debug" | "info" | "warn" | "error" = "none";
    return (level: GGMLLogLevel, message: string) => {
        if (level === "cont" ? levels.some(e => e === prevLevel) : levels.some(e => e === level)) {
            log(message, level === "cont" ? prevLevel : level);
        }
        if (level !== "cont") { prevLevel = level; }
    };
};


const TokenData = struct({ id: Int, logit: Float, p: Float });
const TokenDataArray = struct({
    data: pointer(TokenData),
    size: Size,
    selected: Long,
    sorted: Bool
});
export const ContextParamsStruct = struct({
    n_ctx: Uint,             // text context, 0 = from model
    n_batch: Uint,           // logical maximum batch size that can be submitted to llama_decode
    n_ubatch: Uint,          // physical maximum batch size
    n_seq_max: Uint,         // max number of sequences (i.e. distinct states for recurrent models)
    n_rs_seq: Uint,          // number of recurrent-state snapshots per seq for rollback (0 = no rollback) [EXPERIMENTAL]
    n_outputs_max: Uint,     // max outputs in a ubatch (0 = n_batch)
    n_threads: Int,         // number of threads to use for generation
    n_threads_batch: Int,   // number of threads to use for batch processing

    ctx_type: Int,          // set the context type (e.g. MTP)
    rope_scaling_type: Int, // RoPE scaling type, from `enum llama_rope_scaling_type`
    pooling_type: Int,      // whether to pool (sum) embedding results by sequence id
    attention_type: Int,    // attention type to use for embeddings
    flash_attn_type: Int,   // when to enable Flash Attention

    // ref: https://github.com/ggml-org/llama.cpp/pull/2054
    rope_freq_base: Float,   // RoPE base frequency, 0 = from model
    rope_freq_scale: Float,  // RoPE frequency scaling factor, 0 = from model
    yarn_ext_factor: Float,  // YaRN extrapolation mix factor, negative = from model
    yarn_attn_factor: Float, // YaRN magnitude scaling factor
    yarn_beta_fast: Float,   // YaRN low correction dim
    yarn_beta_slow: Float,   // YaRN high correction dim
    yarn_orig_ctx: Uint,    // YaRN original context size
    defrag_thold: Float,     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)

    cb_eval: Ptr,
    cb_eval_user_data: Ptr,

    type_k: Int, // data type for K cache [EXPERIMENTAL]
    type_v: Int, // data type for V cache [EXPERIMENTAL]

    // Abort callback
    // if it returns true, execution of llama_decode() will be aborted
    // currently works only with CPU execution
    abort_callback: Ptr,
    abort_callback_data: Ptr,

    // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
    embeddings: Bool,  // if true, extract embeddings (together with logits)
    offload_kqv: Bool, // offload the KQV ops (including the KV cache) to GPU
    no_perf: Bool,     // measure performance timings
    op_offload: Bool,  // offload host tensor operations to device
    swa_full: Bool,    // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
    // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13845#issuecomment-2924800573
    kv_unified: Bool,  // use a unified buffer across the input sequences when computing the attention
    // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
    // ref: https://github.com/ggml-org/llama.cpp/pull/14363

    // [EXPERIMENTAL]
    // backend sampler chain configuration (make sure the caller keeps the sampler chains alive)
    // note: the samplers must be sampler chains (i.e. use llama_sampler_chain_init)
    samplers: Ptr,
    n_samplers: Size,

    // a source/target/parent context
    // can be utilized in various ways, for example by sharing results or llama_memory between 2 contexts
    ctx_other: Ptr,
});
export const ModelSplitModes = {
    "none": 0,
    "layer": 1,
    "row": 2,
    "tensor": 3
};
export type ModelSplitMode = keyof typeof ModelSplitModes;
const ModelKVOverrideStruct = struct({
    tag: Int, // enum llama_model_kv_override_type
    key: array(Char, 128),
    value: array(Char, 128)
});
const ModelParamsStruct = struct({
    // NULL-terminated list of devices (ggml_backend_dev_t **)
    devices: pointer(Ptr),                             // ggml_backend_dev_t **
    // NULL-terminated list of buffer type overrides
    tensor_buft_overrides: pointer(struct({ pattern: Str, buft: Ptr })),
    n_gpu_layers: Int,           // number of layers to store in VRAM
    split_mode: Int,             // enum llama_split_mode
    main_gpu: Int,               // GPU used when split_mode == NONE
    tensor_split: pointer(Float),
    progress_callback: pointer(proto(Bool, [Float, Ptr])),
    progress_callback_user_data: Ptr,
    kv_overrides: pointer(ModelKVOverrideStruct),
    vocab_only: Bool,
    use_mmap: Bool,
    use_direct_io: Bool,
    use_mlock: Bool,
    check_tensors: Bool,
    use_extra_bufts: Bool,
    no_host: Bool,
    no_alloc: Bool
});
export const BatchStruct = struct({
    n_tokens: Int,
    token: pointer(Int),
    embd: pointer(Float),
    pos: pointer(Int),
    n_seq_id: pointer(Int),
    seq_id: pointer(pointer(Int)),
    logits: pointer(Byte),
});
export const ModelParamsSchema = z.object({
    progress_callback_interval: z.number(),
    n_gpu_layers: z.number(),
    split_mode: z.enum(["none", "layer", 'row', 'tensor'] as ModelSplitMode[]),
    main_gpu: z.number(),
    tensor_split: z.array(z.number()).min(0).max(Number.POSITIVE_INFINITY),
    vocab_only: z.boolean(),
    use_mmap: z.boolean(),
    use_direct_io: z.boolean(),
    use_mlock: z.boolean(),
    check_tensors: z.boolean(),
    use_extra_bufts: z.boolean(),
    no_host: z.boolean(),
    no_alloc: z.boolean()
}).partial();
export type ModelParamsSerialized = z.output<typeof ModelParamsSchema>;
export interface ModelParams extends ModelParamsSerialized {
    progress_callback?: (progress: number) => boolean | Promise<boolean>,
};
const GGMLTypes = {
    "F64": 28, "F32": 0, "F16": 1, "BF16": 30, "I64": 27, "I32": 26, "I16": 25, "I8": 24,
    "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7, "Q5_K": 13, "Q6_K": 14, "Q8_0": 8, "Q8_1": 9, "Q8_K": 15,
    "IQ1_S": 19, "IQ1_M": 29, "IQ2_XXS": 16, "IQ2_S": 22, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ3_S": 21, "IQ4_NL": 20, "IQ4_XS": 23,
};
export const ContextParamsSchema = z.object({
    n_ctx: z.int().positive(),             // text context, 0 = from model
    n_batch: z.int().positive(),           // logical maximum batch size that can be submitted to llama_decode
    n_ubatch: z.int().positive(),          // physical maximum batch size
    n_seq_max: z.int().positive(),         // max number of sequences (i.e. distinct states for recurrent models)
    n_rs_seq: z.int().positive(),          // number of recurrent-state snapshots per seq for rollback (0 = no rollback) [EXPERIMENTAL]
    n_outputs_max: z.int().positive(),     // max outputs in a ubatch (0 = n_batch)
    n_threads: z.int().positive(),         // number of threads to use for generation
    n_threads_batch: z.int().positive(),   // number of threads to use for batch processing

    ctx_type: z.enum(["default", "mtp"]),          // set the context type (e.g. MTP)
    rope_scaling_type: z.enum(["unspecified", "none", "linear", "yarn", "longrope", "max_value"]), // RoPE scaling type, from `enum llama_rope_scaling_type`
    pooling_type: z.enum(["unspecified", "none", "mean", "cls", "last", "rank"]),      // whether to pool (sum) embedding results by sequence id
    attention_type: z.enum(["unspecified", "casual", "non_casual"]),    // attention type to use for embeddings
    flash_attn_type: z.enum(["auto", "disabled", "enabled"]),   // when to enable Flash Attention

    // ref: https://github.com/ggml-org/llama.cpp/pull/2054
    rope_freq_base: z.number(),   // RoPE base frequency, 0 = from model
    rope_freq_scale: z.number(),  // RoPE frequency scaling factor, 0 = from model
    yarn_ext_factor: z.number(),  // YaRN extrapolation mix factor, negative = from model
    yarn_attn_factor: z.number(), // YaRN magnitude scaling factor
    yarn_beta_fast: z.number(),   // YaRN low correction dim
    yarn_beta_slow: z.number(),   // YaRN high correction dim
    yarn_orig_ctx: z.int().positive(),    // YaRN original context size
    defrag_thold: z.number(),     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)
    type_k: z.enum([
        "F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
        "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16"
    ]),
    type_v: z.enum([
        "F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
        "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16"
    ]),
    // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
    embeddings: z.boolean(),  // if true, extract embeddings (together with logits)
    offload_kqv: z.boolean(), // offload the KQV ops (including the KV cache) to GPU
    no_perf: z.boolean(),     // measure performance timings
    op_offload: z.boolean(),  // offload host tensor operations to device
    swa_full: z.boolean(),    // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
    // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13845#issuecomment-2924800573
    kv_unified: z.boolean(),  // use a unified buffer across the input sequences when computing the attention
    // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
    // ref: https://github.com/ggml-org/llama.cpp/pull/14363
}).partial();
export type ContextParams = z.output<typeof ContextParamsSchema>;
export type SamplerParam = z.output<typeof SamplerParamScheme>;
export const LogitBiasScheme = z.object({ token: z.int(), bias: z.number(), })
export const SamplerParamScheme = z.discriminatedUnion("type", [
    z.object({ type: z.literal("greedy") }),
    z.object({ type: z.literal("dist"), seed: z.int().nonnegative().max(32000) }),
    z.object({ type: z.literal("top_k"), k: z.int().positive() }),
    z.object({ type: z.literal("top_p"), p: z.number(), min_keep: z.int().positive() }),
    z.object({ type: z.literal("min_p"), p: z.number(), min_keep: z.int().positive() }),
    z.object({ type: z.literal("typical"), p: z.number(), min_keep: z.int().positive() }),
    z.object({ type: z.literal("temp"), t: z.number() }),
    z.object({ type: z.literal("temp_ext"), t: z.number(), delta: z.number(), exponent: z.number() }),
    z.object({ type: z.literal("xtc"), p: z.number(), t: z.number(), min_keep: z.int().positive(), seed: z.int().nonnegative().max(32000) }),
    z.object({ type: z.literal("top_n_sigma"), n: z.number() }),
    z.object({ type: z.literal("mirostat"), seed: z.number(), tau: z.number(), eta: z.number(), m: z.number() }),
    z.object({ type: z.literal("mirostat_v2"), seed: z.number(), tau: z.number(), eta: z.number() }),
    z.object({ type: z.literal("grammar"), grammar: z.string(), root: z.string() }),
    z.object({ type: z.literal("grammar_lazy_patterns"), grammar: z.string(), root: z.string(), triggers: z.array(z.union([z.int().nonnegative(), z.string()])) }),
    z.object({ type: z.literal("penalties"), last_n: z.int().positive(), repeat: z.number().optional(), freq: z.number().optional(), present: z.number().optional() }),
    z.object({ type: z.literal("dry"), n_ctx_train: z.number(), multiplier: z.number(), base: z.number(), allowed_length: z.int().nonnegative(), penalty_last_n: z.int().positive(), seq_breakers: z.array(z.string()) }),
    z.object({ type: z.literal("adaptive_p"), target: z.number(), decay: z.number(), seed: z.int().nonnegative().max(32000) }),
    z.object({ type: z.literal("logit_bias"), logit_bias: z.array(LogitBiasScheme) }),
    z.object({ type: z.literal("infill") }),
]);
export const SamplerConstructorScheme = z.array(SamplerParamScheme).min(1);
export type SamplerConstructor = SamplerParam[];
export type BatchConstructor = {
    n_tokens: number,
    token: number[],
    embd: number[] | null,
    pos: number[],
    n_seq_id: number[],
    seq_id: number[][],
    logits: number[],
};













//

