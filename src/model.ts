import z from 'zod';
import { AtomicFlag } from './atomic-flag.js';
import { LibEntropy, LibLlama, LibSamplingHelper, ModelSplitModes, type BatchConstructor, type ContextParams, type GGMLLogLevel, type ModelParams, type ModelParamsSerialized, type SamplerConstructor } from './llama-base.js';
import { expose, getParent } from './worker.js';
import { parentPort } from 'worker_threads';
import { tokeninfoBase, type TokenInfo } from './tokeninfo-base.js';
const { emit, args, exit } = getParent<Events, Args>();


export type API = {
    init: () => bigint | null,
    set_context: (cparams: ContextParams) => void,
    get_n_seq_max: () => number,
    set_sampler: (lineId: number, sampler: SamplerConstructor, offset?: number) => void,
    get_tokens: (lineId: number) => number[],
    get_state: (lineId: number) => { data: Buffer, tokens: number[] },
    set_state: (lineId: number, state: { data: Buffer, tokens: number[] } | null) => void,
    save_state: (lineId: number, file: string) => void,
    load_state: (lineId: number, file: string) => void,
    cancel_input: (lineId: number) => void,
    trim: (lineId: number, nTokens: number) => void,
    push: (lineId: number, content: InputElem[]) => void,
    start: (params: InferenceParams) => void,
    step: (params: InferenceParams) => Generated[] | null,
    metadata: () => Record<string, string>,
    tokeninfo: (start?: number | undefined, end?: number | undefined) => Record<string, TokenInfo>,
    exit: () => void,
};
export type Events = {
    llama_log: [GGMLLogLevel, string],
    progress: [number],
    load_started: [],
    load_cancel: [],
    load_success: [],
    tokens: [Generated[]],
};
export type Args = {
    model_file: string,
    model_params: ModelParamsSerialized,
    llama_library: string,
    entropy_library: string | undefined,
    samplinghelper_library: string | undefined,
    stop_buffer: typeof AtomicFlag.prototype.shared,
};

export type InputElem = {
    text: string,
    special: boolean,
    tokens?: never,
} | {
    text?: never,
    special?: never,
    tokens: number[],
};
export type LineData = {
    lineId: number,
    tokens: number[],
    input: InputElem[],
    sampler: SamplerConstructor,
    samplerPtr: null | bigint,
    samplerOffset: number,
    zeroState: Buffer,
};
export type Generated = (
    {
        lineId: number,
        input: number[],
        stop: boolean,
        stopReasons: StopReason[],
    } & ({
        token: number,
        entropy: number,
    } | {
        token: null,
        entropy: null,
    })
);
export type StopReason = z.output<typeof StopReasonsSchema>;
export const StopReasonsSchema = z.enum(["min_entropy", "max_entropy", "eog_stop", "max_tokens", "manual_stop"]);
export const InferenceLineParamsScheme = z.object({
    min_entropy: z.number().min(0).optional(),
    max_entropy: z.number().min(0).optional(),
    eog_stop: z.boolean().optional(),
    max_tokens: z.int().nonnegative().optional(),
});
export type InferenceLineParams = z.output<typeof InferenceLineParamsScheme>;
export type InferenceParams = {
    line_params: Record<number, InferenceLineParams>
    batch_size_per_line: number,
}


class Instance implements API {
    public readonly llama: LibLlama;
    public readonly entropy: LibEntropy;
    public readonly samplinghelper: LibSamplingHelper;
    public readonly mparams: any;
    public readonly cbInterval: number = 50;
    public readonly stopFlag: AtomicFlag;
    public modelPtr: bigint | null = null;
    public vocabPtr: bigint | null = null;
    public vocabSize: number = 0;
    public contextPtr: bigint | null = null;
    public lines: LineData[] = [];
    public constructor(args: Args) {
        this.llama = new LibLlama(args.llama_library);
        this.llama.log_set((level, message) => emit("llama_log", level, message));
        this.entropy = new LibEntropy(args.entropy_library);
        this.samplinghelper = new LibSamplingHelper(args.samplinghelper_library);
        this.mparams = this.llama.model_default_params();
        for (const key of Object.keys(args.model_params) as (keyof typeof args.model_params)[]) {
            if (key === "progress_callback_interval") {
                const value = args.model_params[key];
                if (value !== undefined) {
                    this.cbInterval = value;
                }
            } else if (key === "split_mode") {
                const value = args.model_params[key];
                if (value !== undefined) {
                    this.mparams[key] = ModelSplitModes[value];
                }
            } else {
                const value = args.model_params[key];
                if (value !== undefined) {
                    this.mparams[key] = value;
                }
            }
        }
        this.stopFlag = new AtomicFlag(args.stop_buffer);
    }
    public init() {
        if (this.modelPtr !== null) { throw new Error(`can use 'init()' only once per worker instance`); }
        let prev = Number.NEGATIVE_INFINITY;
        let hasStarted = false;
        this.mparams["progress_callback" as keyof ModelParams] = (progress: number) => {
            if (!hasStarted) {
                emit("load_started");
                hasStarted = true;
            }
            if (Date.now() - prev >= this.cbInterval) {
                emit("progress", progress);
                prev = Date.now();
            }
            const toContinue = !this.stopFlag.get();
            return toContinue;
        };
        this.modelPtr = this.llama.model_load_from_file(args.model_file, this.mparams);
        if (!hasStarted) {
            emit("load_started");
        }
        if (this.modelPtr === null) {
            emit("load_cancel");
        } else {
            this.vocabPtr = this.llama.model_get_vocab(this.modelPtr);
            this.vocabSize = this.llama.vocab_n_tokens(this.vocabPtr);
            emit("load_success");
        }
        return this.modelPtr;
    }
    public set_context(params: ContextParams) {
        if (this.modelPtr === null) { throw new Error(`model isn't loaded`); }
        if (this.contextPtr !== null) {
            this.llama.context_free(this.contextPtr);
            this.contextPtr = null;
        }
        if (this.mparams["vocab_only"]) { return; }
        const cparams = Object.assign(
            this.llama.context_default_params(),
            this.llama.context_params(params)
        );
        this.contextPtr = this.llama.init_from_model(this.modelPtr, cparams);
        if (this.contextPtr === null) { throw new Error(`cannot load context`); }
        this.lines = Array.from({ length: this.get_n_seq_max() }, (_, i) => ({
            lineId: i,
            tokens: [],
            input: [],
            sampler: [],
            samplerPtr: null,
            samplerOffset: 0,
            zeroState: this.llama.state_seq_get(this.contextPtr as bigint, i)
        }));
    }
    public get_n_seq_max() {
        if (this.mparams["vocab_only"]) { return 0; }
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        return this.llama.n_seq_max(this.contextPtr);
    }
    public set_sampler(lineId: number, sampler: SamplerConstructor, offset: number = 0) {
        if (this.modelPtr === null) { throw new Error(`model isn't loaded`); }
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        line.samplerOffset = offset;
        if (line.samplerPtr !== null) {
            this.llama.sampler_free(line.samplerPtr);
        }
        line.sampler = sampler;
        line.samplerPtr = this.llama.sampler_chain(this.modelPtr, sampler);
        for (const token of line.tokens.slice(offset)) {
            this.llama.sampler_accept(line.samplerPtr, token);
        }
    }
    public get_tokens(lineId: number) {
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        return line.tokens;
    }
    public get_state(lineId: number) {
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        return { data: this.llama.state_seq_get(this.contextPtr, lineId), tokens: line.tokens };
    }
    public set_state(lineId: number, state: { data: Buffer, tokens: number[] } | null) {
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        state = state ?? { data: line.zeroState, tokens: [] };
        this.cancel_input(lineId);
        line.tokens = state.tokens;
        this.llama.state_seq_set(this.contextPtr, lineId, state.data);
        this.set_sampler(lineId, line.sampler);
        emit("tokens", [{ lineId, input: line.tokens, entropy: null, token: null, stop: true, stopReasons: [] }]);
    }
    public save_state(lineId: number, file: string) {
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        this.llama.state_seq_save_file(this.contextPtr, lineId, line.tokens, file);
    }
    public load_state(lineId: number, file: string) {
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        this.cancel_input(lineId);
        line.tokens = [...this.llama.state_seq_load_file(this.contextPtr, lineId, file)];
        this.set_sampler(lineId, line.sampler);
        emit("tokens", [{ lineId, input: line.tokens, entropy: null, token: null, stop: true, stopReasons: [] }]);
    }
    public cancel_input(lineId: number) {
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        line.input = [];
    }
    public trim(lineId: number, nTokens: number) {
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        this.llama.state_seq_trim(this.contextPtr, lineId, nTokens);
        line.tokens = line.tokens.slice(0, line.tokens.length - nTokens);
        this.set_sampler(lineId, line.sampler, line.samplerOffset);
    }
    public push(lineId: number, content: InputElem[]) {
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        for (const elem of content) {
            const last = line.input.at(-1);
            if (last?.special === elem.special && last?.text !== undefined) {
                line.input.pop();
                line.input.push({ text: last.text + elem.text, special: last.special });
            } else {
                line.input.push(elem);
            }
        }
    }
    public start(params: InferenceParams) {
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        let stop = false;
        while (Object.keys(params.line_params).length !== 0 && !stop) {
            params.line_params = Object.fromEntries(Object.entries(params.line_params).flatMap(([lineId, p]) => {
                if (p.max_tokens !== undefined && p.max_tokens < 0) {
                    return [];
                } else {
                    return [[lineId, p]];
                }
            }));
            let generated = this.step(params);
            if (generated === null) { break; }
            generated.forEach(e => {
                const p = params.line_params[e.lineId];
                if (e.token !== null && p?.max_tokens !== undefined) {
                    p.max_tokens--;
                }
            })
            params.line_params = Object.fromEntries(generated.flatMap(e => {
                const p = params.line_params[e.lineId];
                return p !== undefined && !e.stop ? [[e.lineId, p]] : [];
            }));
            stop ||= generated.every(e => e.stop);
            if (!!this.stopFlag.get()) {
                stop = true;
                generated = generated.map(e => Object.assign(
                    Object.assign({}, e),
                    { stop: true, stopReasons: [...e.stopReasons, "manual_stop"] as StopReason[] }
                ));
            }
            emit("tokens", generated);
        }
    }
    public step(params: InferenceParams) {
        const [contextPtr, vocabPtr] = [this.contextPtr, this.vocabPtr];
        if (contextPtr === null) { throw new Error(`context isn't initialized`); }
        if (vocabPtr === null) { throw new Error(`vocab isn't initialized`); }
        const lines = this.lines.flatMap(e => {
            const stop = params.line_params[e.lineId];
            if (stop === undefined) {
                return [];
            } else {
                if (e.samplerPtr === null) {
                    throw new Error(`cannot set line_params for line, that have no sampler`);
                }
                return [{ line: e as LineData & { samplerPtr: bigint }, stop }];
            }
        }).flatMap(e => {
            if (e.line.input.length !== 0) { return [{ line: e.line, stop: e.stop, trimmed: false }]; }
            const token = e.line.tokens.at(-1);
            if (token === undefined) { return []; }
            this.trim(e.line.lineId, 1);
            this.push(e.line.lineId, [{ tokens: [token] }]);
            return [{ line: e.line, stop: e.stop, trimmed: true }];
        }).map(e => {
            let input: number[] = [];
            while (true) {
                const elem = e.line.input.shift();
                if (elem === undefined) { break; }
                input.push(...this.to_tokens(elem));
                if (input.length >= params.batch_size_per_line) {
                    const tail = input.slice(params.batch_size_per_line);
                    input = input.slice(0, params.batch_size_per_line);
                    if (tail.length !== 0) {
                        e.line.input.unshift({ tokens: tail });
                    }
                    break;
                }
            }
            return { line: e.line, stop: e.stop, input, logits: e.line.input.length === 0, trimmed: e.trimmed };
        });
        const batch: BatchConstructor = {
            n_tokens: lines.map(e => e.input.length).reduce((a, b) => a + b, 0),
            token: lines.flatMap(e => e.input),
            pos: lines.flatMap(e => e.input.map((t, i) => e.line.tokens.length + i)),
            n_seq_id: lines.flatMap(e => e.input.map(() => 1)),
            seq_id: lines.flatMap(e => e.input.map(() => [e.line.lineId])),
            embd: null,
            logits: lines.flatMap(e => e.input.map((t, i) => i === e.input.length - 1 && e.logits ? 1 : 0)),
        };
        if (batch.n_tokens === 0) { return null; }
        const error = this.llama.decode(contextPtr, batch);
        if (error !== 0) {
            throw new Error(`llama_decode error ${error}`);
        }
        const on_logits = (e: typeof lines extends (infer E)[] ? E : never) => {
            const line_index = lines.findIndex(ee => ee === e);
            if (line_index < 0) { throw new Error(`unexpected situation: cannot find line, that's currently processing`); }
            const logit_index = lines.slice(0, line_index + 1).map(e => e.input.length).reduce((a, b) => a + b, 0) - 1;

            const logits = this.llama.get_logits_ith(contextPtr, this.vocabSize, logit_index);
            const cur_p = this.samplinghelper.logitsToCurp(logits);
            this.llama.sampler_apply(e.line.samplerPtr, cur_p);
            const token = this.samplinghelper.curpToToken(cur_p);
            const entropy = this.entropy.entropyOfLogits(logits);

            const stopReasons: Generated["stopReasons"] = [];
            if ((e.stop.eog_stop ?? false) && this.llama.vocab_is_eog(vocabPtr, token)) { stopReasons.push("eog_stop"); }
            if (entropy > (e.stop.max_entropy ?? Number.POSITIVE_INFINITY)) { stopReasons.push("max_entropy"); }
            if (entropy < (e.stop.min_entropy ?? 0)) { stopReasons.push("min_entropy"); }
            if ((e.stop.max_tokens ?? Number.POSITIVE_INFINITY) <= 1) { stopReasons.push("max_tokens"); }
            return { token, entropy, stopReasons };
        }
        const push_tokens = (input: number[], e: typeof lines extends (infer E)[] ? E : never) => {
            const offset = e.line.samplerOffset - e.line.tokens.length;
            input.slice(offset < 0 ? 0 : offset).forEach(ee => this.llama.sampler_accept(e.line.samplerPtr, ee));
            e.line.tokens.push(...e.input);
        };
        return lines.map(e => {
            let input: number[];
            let token: number | null;
            let entropy: number | null;
            let stopReasons: Generated["stopReasons"];
            if (e.trimmed) {
                input = [];
                if (e.logits) {
                    push_tokens(input, e);
                    const res = on_logits(e);
                    token = res.token;
                    entropy = res.entropy;
                    stopReasons = res.stopReasons;
                    this.push(e.line.lineId, [{ tokens: [token] }]);
                } else {
                    token = null;
                    entropy = null;
                    stopReasons = [];
                }
            } else {
                input = e.input;
                if (e.logits) {
                    push_tokens(input, e);
                    const res = on_logits(e);
                    token = res.token;
                    entropy = res.entropy;
                    stopReasons = res.stopReasons;
                    this.push(e.line.lineId, [{ tokens: [token] }]);
                } else {
                    push_tokens(input, e);
                    token = null;
                    entropy = null;
                    stopReasons = [];
                }
            }
            return { lineId: e.line.lineId, input, stop: stopReasons.length !== 0, stopReasons, token, entropy } as Generated;
        });
    }
    public to_tokens(elem: InputElem) {
        if (elem.tokens !== undefined) { return elem.tokens; }
        if (this.vocabPtr === null) {
            throw new Error(`cannot call to_tokens on text-elems without initialized vocab`);
        }
        return this.llama.tokenize(this.vocabPtr, elem.text, elem.special);
    }
    public metadata() {
        if (this.modelPtr === null) { throw new Error(`model isn't loaded`); }
        const result: Record<string, string> = {};
        const count = this.llama.model_meta_count(this.modelPtr);
        for (let i = 0; i < count; i++) {
            const key = this.llama.model_meta_key_by_index(this.modelPtr, i);
            const value = this.llama.model_meta_val_str_by_index(this.modelPtr, i);
            result[key] = value;
        }
        return result;
    }
    public tokeninfo(start: number = 0, end: number = Number.POSITIVE_INFINITY) {
        if (this.vocabPtr === null) {
            throw new Error(`vocab must be loaded to call .tokeninfo()`);
        }
        const startIndex = start <= 0 ? 0 : start >= this.vocabSize ? this.vocabSize : start;
        const endIndex = end <= 0 ? 0 : end >= this.vocabSize ? this.vocabSize : end;
        return tokeninfoBase(this.llama, this.vocabPtr, startIndex, endIndex);
    }
    public free() {
        if (this.contextPtr !== null) {
            this.llama.context_free(this.contextPtr);
            this.contextPtr = null;
        }
        if (this.modelPtr !== null) {
            this.llama.model_free(this.modelPtr);
            this.modelPtr = null;
        }
    }
    public exit() {
        this.free();
        exit();
    }
}
if (parentPort !== null) {
    const instance = new Instance(args);
    expose<API>({
        init: () => instance.init(),
        exit: () => instance.exit(),
        set_context: (cparams) => instance.set_context(cparams),
        get_n_seq_max: () => instance.get_n_seq_max(),
        set_sampler: (ln, s, o = 0) => instance.set_sampler(ln, s, o),
        get_tokens: (ln) => instance.get_tokens(ln),
        get_state: (ln) => instance.get_state(ln),
        set_state: (ln, st) => instance.set_state(ln, st),
        save_state: (ln, file) => instance.save_state(ln, file),
        load_state: (ln, file) => instance.load_state(ln, file),
        cancel_input: (ln) => instance.cancel_input(ln),
        trim: (ln, n) => instance.trim(ln, n),
        push: (ln, c) => instance.push(ln, c),
        start: (p) => instance.start(p),
        step: (p) => instance.step(p),
        metadata: () => instance.metadata(),
        tokeninfo: (s = 0, e = undefined) => instance.tokeninfo(s, e),
    });
}



//
