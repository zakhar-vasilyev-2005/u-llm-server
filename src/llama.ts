import fs from 'fs-extra';
import { EventEmitter } from 'events';
import path from 'path';
import { createFreeEvent } from './event-util.js';
import { Worker } from './worker.js';
import type { API as modelAPI, Events as modelEvents, Args as modelArgs, InferenceParams, InferenceLineParams, StopReason } from './model.js';
import { AtomicFlag } from './atomic-flag.js';
import { createGGMLLogger, LibEntropy, LibGGML, LibLlama, LibSamplingHelper, type ContextParams, type GGMLLogLevel, type ModelParams, type ModelParamsSerialized, type SamplerConstructor } from './llama-base.js';
import { GrowBuffer } from './growbuffer.js';
import * as tmp from 'tmp';



export type BackendEvents = {
    backend_free: [];
};
export type BackendLogParams = {
    ggml_levels?: GGMLLogLevel[],
    llama_levels?: GGMLLogLevel[],
    ggml?: boolean | ((message: string, level: "none" | "debug" | "info" | "warn" | "error") => void),
    llama?: boolean | ((message: string, level: "none" | "debug" | "info" | "warn" | "error") => void),
};
export class Backend extends EventEmitter<BackendEvents> {
    public static fromPath(
        folder: string,
        entropyFolder: string | undefined = undefined,
        samplinghelperFolder: string | undefined = undefined,
        logParams: BackendLogParams = {}
    ) {
        if (!fs.existsSync(folder)) { throw new Error(`requested folder doesn't exist (path: ${JSON.stringify(folder)})`); }
        const ggml = new LibGGML(path.join(folder, "libggml.so"));
        const llama = new LibLlama(path.join(folder, "libllama.so"));
        const entropy = new LibEntropy(entropyFolder === undefined ? undefined : path.join(entropyFolder, "libentropy.so"));
        const samplinghelper = new LibSamplingHelper(samplinghelperFolder === undefined ? undefined : path.join(samplinghelperFolder, "libsamplinghelper.so"));
        return new Backend(llama, ggml, entropy, samplinghelper, folder, logParams);
    }
    public constructor(
        public readonly llama: LibLlama,
        public readonly ggml: LibGGML,
        public readonly entropy: LibEntropy,
        public readonly samplinghelper: LibSamplingHelper,
        public readonly backend_path: string,
        logParams: BackendLogParams = {}
    ) {
        super();
        this.log(logParams);
        ggml.backend_load_all_from_path(backend_path);
        llama.backend_init();
        llama.on("lib_free", () => this.free());
        ggml.on("lib_free", () => this.free());
    }
    public log(logParams: BackendLogParams) {
        const getLog = (source: string, log: typeof logParams.llama) => (
            log === true
                ? (s: string, l: GGMLLogLevel) => console.log(`${source} ${l}`, s.trimEnd())
                : log ? log : () => { }
        );
        this.ggml.log_set(createGGMLLogger(logParams.ggml_levels ?? ["info", "warn", "error"], getLog("GGML", logParams.ggml)));
        this.llama.log_set(createGGMLLogger(logParams.llama_levels ?? ["info", "warn", "error"], getLog("LLAMA", logParams.llama)));
    }
    public readonly free = createFreeEvent<this, void>("backend_free", () => this.llama.backend_free());
}

export type TokenData = {
    token: number,
    piece: string,
    piece_raw: Buffer,
    control: boolean,
};
export type TokenGenerated = {
    line: ModelLine,
    next: TokenData | null,
    input: TokenData[],
    entropy: number,
    stop: boolean,
    stopReasons: StopReason[],
};

export type ModelEvents = {
    llama_log: [GGMLLogLevel, string],
    tokens: [TokenGenerated[]],
    model_free: [],
    generation_started: [],
    generation_stopped: [],
}
export class Model extends EventEmitter<ModelEvents> {
    public static async load(
        backend: Backend,
        model_file: string,
        model_params: ModelParams,
        context_params: ContextParams,
        batchSizePerLine: number = 64,
    ) {
        model_params = Object.assign({}, model_params);
        context_params = Object.assign({}, context_params);
        const progressCallback = model_params.progress_callback ?? (() => true);
        delete model_params.progress_callback;
        if (!await fs.exists(model_file)) {
            throw new Error(`given model file not exists: ${JSON.stringify(model_file)}`);
        }
        const stopFlag = new AtomicFlag();
        stopFlag.set(0);
        const workerFile = path.join(import.meta.dirname, "model.js");
        const worker = await Worker.start<modelAPI, modelEvents, modelArgs>(workerFile, {
            llama_library: backend.llama.file,
            entropy_library: backend.entropy.raw?.file,
            samplinghelper_library: backend.samplinghelper.raw?.file,
            model_file,
            model_params: model_params as ModelParamsSerialized,
            stop_buffer: stopFlag.shared,
        });
        let progress = -1;
        worker.on("progress", p => { progress = p; });
        let loop = true;
        let loaded = false;
        worker.once("load_success", () => { loop = false; loaded = true; });
        worker.once("load_cancel", () => { loop = false; });
        let modelPtr: bigint | null = null;
        await Promise.all([
            new Promise(async resolve => {
                let prev = Number.NEGATIVE_INFINITY;
                while (loop) {
                    if (prev < progress && progress >= 0) {
                        prev = progress;
                        const ok = await progressCallback(progress);
                        if (!ok) { stopFlag.set(1); }
                    }
                    await new Promise(resolve => setTimeout(resolve, model_params.progress_callback_interval ?? 50));
                }
                if (prev < 1) {
                    prev = 1;
                    const ok = await progressCallback(1);
                    if (!ok) { stopFlag.set(1); }
                }
                resolve(undefined);
            }),
            worker.api.init().then(model_ptr => { modelPtr = model_ptr }),
        ]);
        if (!loaded || modelPtr === null) { throw Object.assign(new Error(`model not loaded`), { code: "MODEL_NOT_LOADED" }); }
        const vocabPtr = backend.llama.model_get_vocab(modelPtr);
        const metadata = await worker.api.metadata();
        await worker.api.set_context(context_params);
        const n_seq_max = await worker.api.get_n_seq_max();
        const model = new Model(
            backend, worker, stopFlag, model_file, modelPtr, vocabPtr,
            Object.freeze(model_params), Object.assign(context_params, { n_seq_max }),
            Object.freeze(metadata), batchSizePerLine
        );
        await Promise.all(model.lines.map(e => e.setSampler([{ type: "greedy" }])));
        return model;
    }
    public readonly lines: readonly ModelLine[];
    public constructor(
        public readonly backend: Backend,
        public readonly worker: Worker<modelAPI, modelEvents>,
        public readonly stopFlag: AtomicFlag,
        public readonly modelFile: string,
        public readonly modelPtr: bigint,
        public readonly vocabPtr: bigint,
        public readonly modelParams: ModelParamsSerialized,
        public contextParams: ContextParams & { n_seq_max: number },
        public readonly metadata: Record<string, string>,
        public readonly batchSizePerLine: number,
    ) {
        super();
        const tokenToPiece = (line: ModelLine, token: number) => {
            let piece_raw: Buffer | null = null;
            let piece: string | null = null;
            if (token !== null) {
                piece_raw = this.backend.llama.token_to_piece_raw(this.vocabPtr, token);
                line.buffer.push(piece_raw)
                const buffer = line.buffer.buffer;
                const raw = buffer.toString();
                if (raw.includes("\ufffe")) {
                    let len = buffer.byteLength;
                    while (buffer.toString("utf8", 0, len).endsWith("\ufffe")) {
                        len--;
                    }
                    piece = buffer.toString("utf8", 0, len);
                    line.buffer.replaceWith(buffer.subarray(len));
                } else {
                    line.buffer.clear();
                    piece = raw;
                }
            }
            const control = token === null ? false : this.backend.llama.vocab_is_control(this.vocabPtr, token);
            return { token, piece, piece_raw, control };
        }
        this.backend.llama.log_set((lv, m) => this.emit("llama_log", lv, m));
        this.worker.on("llama_log", (lv, m) => this.emit("llama_log", lv, m));
        this.worker.on("tokens", raw => {
            const tokens = raw.map(e => {
                const line = this.lines[e.lineId] as ModelLine;
                const input = e.input.map(e => tokenToPiece(line, e));
                const temp = line.buffer.copy();
                const next = e.token === null ? null : tokenToPiece(line, e.token);
                line.buffer.replaceWith(temp);
                const { entropy, stop, stopReasons } = e;
                return { line, next, input, entropy, stop, stopReasons } as TokenGenerated;
            });
            this.emit("tokens", tokens);
            tokens.forEach(e => {
                e.line.content.push(...e.input);
                e.line.emit("token", e);
            })
        })
        this.lines = Object.freeze(Array.from({ length: contextParams.n_seq_max }, (e, i) => new ModelLine(this, i)));
    }
    public async setContext(contextParams: ContextParams) {
        await this.worker.api.set_context(contextParams);
        const n_seq_max = await this.worker.api.get_n_seq_max();
        this.contextParams = Object.assign(contextParams, { n_seq_max });
    }
    protected generating: Promise<void> | null = null;
    public isGenerating() { return this.generating !== null; }
    public start() {
        this.stopFlag.set(0);
        this.emit("generation_started");
        this.generating = this.worker.api.start({
            line_params: this.lines.filter(e => e.enabled).map(e => e.inferenceParams),
            batch_size_per_line: this.batchSizePerLine,
        }).then(() => {
            this.emit("generation_stopped");
        });
    }
    public async stop() {
        if (this.generating === null) { return; }
        this.stopFlag.set(1);
        await this.generating;
        this.generating = null;
    }
    public tokenize(text: string, parseSpecial: boolean = true, addSpecial: boolean = false) {
        return this.backend.llama.tokenize(this.vocabPtr, text, parseSpecial, addSpecial);
    }
    public detokenize(tokens: number[], unparseSpecial: boolean = true, removeSpecial: boolean = false) {
        return this.backend.llama.detokenize(this.vocabPtr, tokens, unparseSpecial, removeSpecial).toString("utf8");
    }
    public readonly free = createFreeEvent<this, Promise<void>>("model_free", async () => {
        await this.stop();
        await this.worker.api.exit();
    });
}

export type LineEvents = {
    token: [TokenGenerated],
};
export class ModelLine extends EventEmitter<LineEvents> {
    public buffer = new GrowBuffer();
    public content: TokenData[] = [];
    public constructor(public readonly model: Model, public readonly index: number) {
        super();
    }
    public async pushInput(input: number[]): Promise<void>;
    public async pushInput(input: string, parseSpecial?: boolean): Promise<void>;
    public async pushInput(input: number[] | string, parseSpecial: boolean = true) {
        await this.model.worker.api.push(this.index, input, parseSpecial);
    }
    public async cancelInput() {
        await this.model.worker.api.cancel_input(this.index);
    }
    public async setSampler(sampler: SamplerConstructor, samplerOffset: number = 0) {
        await this.model.worker.api.set_sampler(this.index, sampler, samplerOffset);
    }
    public async trim(nTokens: number) {
        await this.model.worker.api.trim(this.index, nTokens);
        this.content = this.content.slice(0, this.content.length - nTokens);
    }
    public async start() {
        await this.model.stop();
        this.enabled = true;
        this.model.start();
    }
    public async stop() {
        await this.model.stop();
        this.enabled = false;
        if (this.model.lines.some(e => e.enabled)) {
            this.model.start();
        }
    }
    public async setState(state: { data: Buffer, tokens: number[] } | null) {
        await this.model.worker.api.set_state(this.index, state);
    }
    public async getState(): Promise<{ data: Buffer, tokens: number[] }> {
        return await this.model.worker.api.get_state(this.index);
    }
    public async getTokens(): Promise<number[]> {
        return await this.model.worker.api.get_tokens(this.index);
    }
    public async clearState() {
        await this.model.worker.api.set_state(this.index, null);
    }
    public async loadState(file: string) {
        await this.model.worker.api.load_state(this.index, file);
    }
    public async saveState(file: string | undefined) {
        if (file === undefined) {
            file = await new Promise((resolve, reject) => tmp.tmpName({ dir: "/dev/shm", postfix: ".bin" }, (err, name) => err ? reject(err) : resolve(name)));
        }
        if (file === undefined) {
            throw new Error(`cannot create temporary file name`);
        }
        await this.model.worker.api.save_state(this.index, file);
        return file;
    }
    public enabled: boolean = false;
    public inferenceParams: InferenceLineParams = {
        min_entropy: 0,
        max_entropy: Number.POSITIVE_INFINITY,
        eog_stop: true,
    };
}









