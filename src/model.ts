import z from 'zod';
import { AtomicFlag } from './atomic-flag.js';
import { LibEntropy, LibLlama, LibSamplingHelper, ModelSplitModes, type BatchConstructor, type ContextParams, type GGMLLogLevel, type ModelParams, type ModelParamsSerialized, type SamplerConstructor } from './llama-base.js';
import { expose, getParent } from './worker.js';
import { parentPort } from 'worker_threads';
const { emit, args, exit } = getParent<Events, Args>();


export type API = {
    init: () => bigint | null,
    set_context: (cparams: ContextParams) => void,
    get_n_seq_max: () => number,
    line_set_sampler: (lineId: number, sampler: SamplerConstructor) => void,
    line_get_tokens: (lineId: number) => number[],
    line_get_state: (lineId: number) => { data: Buffer, tokens: number[] },
    line_set_state: (lineId: number, state: { data: Buffer, tokens: number[] } | null) => void,
    line_save_state: (lineId: number, file: string) => void,
    line_load_state: (lineId: number, file: string) => void,
    line_cancel_input: (lineId: number) => void,
    trim: (lineId: number, nTokens: number) => void,
    push: (lineId: number, content: string | number[], parseSpecial?: boolean) => void,
    start: (params: InferenceParams) => void,
    step: (params: InferenceParams) => Generated[] | null,
    metadata: () => Record<string, string>,
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

export type LineData = {
    lineId: number,
    tokens: number[],
    input: number[],
    sampler: SamplerConstructor,
    samplerPtr: null | bigint,
    zeroState: Buffer,
};
export type Generated = (
    {
        lineId: number,
        input: number[],
        stop: boolean,
        stopReasons: StopReason[],
        replace: boolean,
    } & ({
        token: number,
        entropy: number,
    } | {
        token: null,
        entropy: null,
    })
);
export type StopReason = z.output<typeof StopReasonsSchema>;
export const StopReasonsSchema = z.enum(["min_entropy", "max_entropy", "eog_stop", "max_tokens"]);
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
            zeroState: this.llama.state_seq_get(this.contextPtr as bigint, i)
        }));
    }
    public get_n_seq_max() {
        if (this.mparams["vocab_only"]) { return 0; }
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        return this.llama.n_seq_max(this.contextPtr);
    }
    public line_set_sampler(lineId: number, sampler: SamplerConstructor) {
        if (this.modelPtr === null) { throw new Error(`model isn't loaded`); }
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        if (line.samplerPtr !== null) {
            this.llama.sampler_free(line.samplerPtr);
        }
        line.sampler = sampler;
        line.samplerPtr = this.llama.sampler_chain(this.modelPtr, sampler);
        for (const token of line.tokens) {
            this.llama.sampler_accept(line.samplerPtr, token);
        }
    }
    public line_get_tokens(lineId: number) {
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        return line.tokens;
    }
    public line_get_state(lineId: number) {
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        return { data: this.llama.state_seq_get(this.contextPtr, lineId), tokens: line.tokens };
    }
    public line_set_state(lineId: number, state: { data: Buffer, tokens: number[] } | null) {
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        state = state ?? { data: line.zeroState, tokens: [] };
        this.line_cancel_input(lineId);
        line.tokens = state.tokens;
        this.llama.state_seq_set(this.contextPtr, lineId, state.data);
        this.line_set_sampler(lineId, line.sampler);
        emit("tokens", [{ lineId, input: line.tokens, entropy: null, token: null, replace: false, stop: true, stopReasons: [] }]);
    }
    public line_save_state(lineId: number, file: string) {
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        this.llama.state_seq_save_file(this.contextPtr, lineId, line.tokens, file);
    }
    public line_load_state(lineId: number, file: string) {
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        this.line_cancel_input(lineId);
        line.tokens = [...this.llama.state_seq_load_file(this.contextPtr, lineId, file)];
        this.line_set_sampler(lineId, line.sampler);
        emit("tokens", [{ lineId, input: line.tokens, entropy: null, token: null, replace: false, stop: true, stopReasons: [] }]);
    }
    public line_cancel_input(lineId: number) {
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
    }
    public push(lineId: number, content: string | number[], parseSpecial: boolean = true) {
        const line = this.lines[lineId];
        if (line === undefined) { throw new Error(`line #${lineId} not found`); }
        if (typeof content === "string") {
            if (this.vocabPtr === null) {
                throw new Error(`model's vocab isn't initialized`);
            }
            line.input.push(...this.llama.tokenize(this.vocabPtr, content, parseSpecial));
        } else {
            line.input.push(...content);
        }
    }
    public start(params: InferenceParams) {
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        while (Object.keys(params.line_params).length !== 0) {
            params.line_params = Object.fromEntries(Object.entries(params.line_params).flatMap(([lineId, p]) => {
                return p.max_tokens !== undefined && p.max_tokens <= 0 ? [] : [[lineId, p]];
            }));
            const generated = this.step(params);
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
            emit("tokens", generated);
            if (this.stopFlag.get() || generated.every(e => e.stop)) { break; }
        }
    }
    public step(params: InferenceParams) {
        if (this.contextPtr === null) { throw new Error(`context isn't initialized`); }
        const trimmedLines: number[] = [];
        let batchTokens = 0;
        const batchData = this.lines.flatMap(line => {
            const lineConfig = params.line_params[line.lineId];
            if (lineConfig === undefined) { return []; }
            if (line.samplerPtr === null) {
                return [{
                    line,
                    startPos: 0,
                    input: [],
                    logitIndex: null,
                    lineConfig
                }];
            }
            let input: number[];
            while (true) {
                if (line.input.length === 0 && line.tokens.length >= 1) {
                    trimmedLines.push(line.lineId);
                    const token = line.tokens.at(-1) as number;
                    this.trim(line.lineId, 1);
                    this.push(line.lineId, [token]);
                    continue;
                } else {
                    input = line.input.slice(0, params.batch_size_per_line);
                    line.input = line.input.slice(params.batch_size_per_line);
                    break;
                }
            }
            batchTokens += input.length;
            const notEnd = line.input.length > 0;
            return [{
                line,
                startPos: line.tokens.length,
                input,
                logitIndex: notEnd || input.length === 0 ? null : batchTokens - 1,
                lineConfig,
            }];
        });
        if (batchTokens === 0) { return null; }
        const batch: BatchConstructor = (() => {
            const token = Array(batchTokens);
            const pos = Array(batchTokens);
            const seq_id = Array(batchTokens);
            const logits = Array(batchTokens).fill(0);
            for (const seqData of batchData) {
                const lineId = seqData.line.lineId;
                const { startPos, input } = seqData;
                for (let i = 0; i < input.length; i++) {
                    token[i] = input[i];
                    pos[i] = startPos + i;
                    seq_id[i] = [lineId];
                }
                if (seqData.logitIndex !== null) {
                    logits[seqData.logitIndex] = 1;
                }
            }
            return { n_tokens: batchTokens, token, embd: null, pos, n_seq_id: Array(batchTokens).fill(1), seq_id, logits, };
        })();
        const code = this.llama.decode(this.contextPtr, batch);
        if (code !== 0) { throw new Error(`llama_decode error code ${code}`); }
        const generated = batchData.map(seqData => {
            const { lineId } = seqData.line;
            const { input, lineConfig } = seqData;
            (this.lines[lineId] as LineData).tokens.push(...input);
            let entropy = 0;
            let token: number | null = null;
            if (seqData.logitIndex !== null && seqData.line.samplerPtr !== null) {
                const logits = this.llama.get_logits_ith(this.contextPtr as bigint, this.vocabSize, seqData.logitIndex);
                entropy = this.entropy.entropyOfLogits(logits);
                const cur_p = this.samplinghelper.logitsToCurp(logits);
                this.llama.sampler_apply(seqData.line.samplerPtr, cur_p);
                token = this.samplinghelper.curpToToken(cur_p);
                this.push(lineId, [token]);
            }
            const stopReasons: StopReason[] = [];
            if (entropy < (lineConfig.min_entropy ?? 0)) {
                stopReasons.push("min_entropy");
            }
            if (entropy > (lineConfig.max_entropy ?? Number.POSITIVE_INFINITY)) {
                stopReasons.push("max_entropy");
            }
            if ((lineConfig.eog_stop ?? false) && token !== null && this.llama.vocab_is_eog(this.vocabPtr as bigint, token)) {
                stopReasons.push("eog_stop");
            }
            if (lineConfig.max_tokens !== undefined && lineConfig.max_tokens <= 1) {
                stopReasons.push("max_tokens");
            }
            return { lineId, token, entropy, input, replace: trimmedLines.some(e => e === lineId), stop: stopReasons.length > 0, stopReasons } as Generated;
        });
        return generated;
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
        line_set_sampler: (ln, s) => instance.line_set_sampler(ln, s),
        line_get_tokens: (ln) => instance.line_get_tokens(ln),
        line_get_state: (ln) => instance.line_get_state(ln),
        line_set_state: (ln, st) => instance.line_set_state(ln, st),
        line_save_state: (ln, file) => instance.line_save_state(ln, file),
        line_load_state: (ln, file) => instance.line_load_state(ln, file),
        line_cancel_input: (ln) => instance.line_cancel_input(ln),
        trim: (ln, n) => instance.trim(ln, n),
        push: (ln, c, sp) => instance.push(ln, c, sp),
        start: (p) => instance.start(p),
        step: (p) => instance.step(p),
        metadata: () => instance.metadata(),
    });
}



//