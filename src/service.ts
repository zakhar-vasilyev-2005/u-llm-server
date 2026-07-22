import path from "path";
import { Backend, ModelLine, Model } from "./llama.js";
import { type ContextParams, type ModelParamsSerialized, type SamplerConstructor } from "./llama-base.js";
import { SResultArgs, SCommandArgs, SEventArgs, SCommandSchema, SErrorArgs, SMessageSchema, type SCommand, type SError, type SEvent, type SMessage, type SResult, SToken } from './server-schemas.js';
import { createConnection, createServer, Server, Socket, type NetConnectOpts } from "net";
import { EventEmitter } from "events";
import * as z from "zod";
import type { Serializable } from "./serializable.js";
import { sprintf } from "sprintf-js";
import { Yurandom } from 'yurandom';
import { createFreeEvent } from "./event-util.js";
import { extractMiddle } from "./extract-middle.js";
import type { InferenceLineParams, StopReason } from "./model.js";
import { Template } from "@huggingface/jinja";
import { fork, type IOType } from "child_process";
import type Stream from "stream";



type PromiseOrNot<T> = T | Promise<T>;
export type ConnOption = { unix: string } | { host?: string, port: number };


export const runModelDefaults = {
    modelParams: {
        n_gpu_layers: 999,
        check_tensors: false,
        use_extra_bufts: true,
    } as ModelParamsSerialized,
    contextParams: {
        n_ctx: 2048,
        n_batch: 256,
        n_ubatch: 256,
        n_seq_max: 1,
        embeddings: false,
        kv_unified: true,
        offload_kqv: true,
        no_perf: true,
        type_k: "Q8_0",
        type_v: "Q8_0",
    } as ContextParams,
};
export type RunModelParams = {
    binaries_path: string,
    model_file: string,
    callback: (model: Model) => void | Promise<void>,
    log: (message: string) => void,
};
export async function runModel(params: RunModelParams) {
    const log = params.log ?? (() => { });
    const backend = Backend.fromPath(
        path.join(params.binaries_path, "llama-b9844"),
        path.join(params.binaries_path, "utils"),
        path.join(params.binaries_path, "utils"),
        {
            ggml_levels: ["none", "debug", "info", "warn", "error", "cont"],
            llama_levels: ["none", "debug", "info", "warn", "error", "cont"],
            ggml: (msg, lv) => log(`GGML ${lv.toUpperCase()}: ${msg}`),
            llama: (msg, lv) => log(`LLAMA ${lv.toUpperCase()}: ${msg}`),
        }
    );
    try {
        const preloadStart = Date.now();
        let loadStart = Date.now();
        let started = false;
        let prevProgress = 0;
        const model_params = Object.assign(runModelDefaults.modelParams, {
            progress_callback: async (progress: number) => {
                if (!started) {
                    log(`PRELOAD: done in ${Date.now() - preloadStart}ms`);
                    log(`MODEL: ${sprintf("%9.5f", 0)}%`);
                    loadStart = Date.now();
                    started = true;
                }
                if (progress > prevProgress) {
                    log(`MODEL: ${sprintf("%9.5f", progress * 100)}%`);
                    prevProgress = progress;
                }
                return true;
            }
        });
        const model = await Model.load(backend, params.model_file, model_params, runModelDefaults.contextParams);
        try {
            log(`MODEL: done in ${Date.now() - loadStart}ms, model file: ${JSON.stringify(model.modelFile)}`);
            await params.callback(model);
        } finally {
            await model.free();
        }
    } finally {
        backend.free();
    }
}



export type ModelServerEvents = {
    command: [SCommand, Socket],
    model_loaded: [],
    close: [],
};
export class ModelServer extends EventEmitter<ModelServerEvents> {
    public server: Server;
    public socketIds = new WeakMap<Socket, number>();
    public socketCount = 0;
    public sockets: Socket[] = [];
    public model: Model | null = null;
    public freeLines: ModelLine[] = [];
    public activeLines: Record<string, ModelLine> = {};
    public constructor(public readonly modelFile: string, public readonly modelParams: ModelParamsSerialized) {
        super();
        this.setMaxListeners(100);
        this.server = createServer(socket => {
            if (this.server !== undefined) {
                this.socketIds.set(socket, this.socketCount++);
                this.log(`CONNECT (client ${this.socketIds.get(socket)})`);
                this.sockets.push(socket);
                let buffer = "";
                socket.on("data", data => {
                    buffer += data.toString();
                    const parts = buffer.split("\n");
                    if (parts.length > 1) {
                        buffer = parts.pop() as string;
                        for (const part of parts) {
                            let raw: Serializable;
                            try { raw = JSON.parse(part); }
                            catch (e) {
                                this.send(socket, {
                                    type: "event",
                                    object: { event: "command_json_error", args: { message: String(e) } }
                                });
                                continue;
                            }
                            const parsed = z.safeParse(SCommandSchema, raw);
                            if (parsed.success) {
                                this.log(`COMMAND CALL (client ${this.socketIds.get(socket)}): ${JSON.stringify(parsed.data)}`);
                                this.emit("command", parsed.data, socket);
                            } else {
                                this.send(socket, {
                                    type: "event",
                                    object: {
                                        event: "command_schema_error",
                                        args: {
                                            message: parsed.error.message,
                                            issues: parsed.error.issues
                                        }
                                    }
                                })
                            }
                        }
                    }
                });
                socket.on("close", () => {
                    this.log(`DISCONNECT (client ${this.socketIds.get(socket)})`);
                    this.sockets = this.sockets.filter(s => s !== socket);
                });
            }
        });
        this.server.on("close", () => {
            this.log(`SERVER CLOSE`);
        });
        this.initCommands();
    }
    public initCommands() {
        const server = this;
        function getModel() {
            if (server.model === null) {
                throw new Error(`cannot get model object from server: server not initialized`);
            }
            return server.model;
        }
        server.bind("start", async () => {
            while (this.model === null) {
                await new Promise(resolve => setTimeout(resolve, 50));
            }
            return { command: "start", args: { model_params: getModel().modelParams, metadata: getModel().metadata } };
        });
        server.bind("set_context", async args => {
            await getModel().setContext(args.context_params);
            this.freeLines = [...this.model?.lines ?? []];
            this.activeLines = {};
            return [{ event: "ctx_changed", args: { context_params: args.context_params } }, { command: "set_context", args: null }];
        });
        server.bind("line_init", async args => {
            const newLineId = () => {
                while (true) {
                    const line_id = "ln_" + new Yurandom(`${process.pid}_${Date.now()}`).hex(2).toLowerCase();
                    if (line_id in this.activeLines) { continue; }
                    return line_id;
                }
            }
            const line_id = args.line_id ?? newLineId();
            let line = this.activeLines[line_id];
            if (line === undefined) {
                line = this.freeLines.pop();
                if (line === undefined) {
                    const max_lines = getModel().contextParams.n_seq_max;
                    const message = `Cannot create new line: too many lines already exist. Current maximum is ${max_lines}.`;
                    return { error: "too_many_lines", args: { message, fields: { max_lines } } };
                }
                this.activeLines[line_id] = line;
            }
            if (args.sampler !== undefined) {
                await line.setSampler(args.sampler);
            }
            if (args.inference !== undefined) {
                line.inferenceParams = args.inference;
            }
            return { command: "line_init", args: { line_id } };
        });
        server.bind("line_free", args => {
            const line = this.activeLines[args.line_id];
            if (line === undefined) {
                const message = `Cannot find line with id ${JSON.stringify(args.line_id)}: no such id in the list.`;
                return { error: "line_not_found", args: { message, fields: { line_id: args.line_id } } };
            } else {
                this.freeLines.push(line);
                delete this.activeLines[args.line_id];
                return { command: "line_free", args: null };
            }
        });
        server.bind("line_list", () => {
            return { command: "line_list", args: Object.keys(this.activeLines).map(e => ({ line_id: e })) };
        });
        server.bind("line_load", async args => {
            const line = this.activeLines[args.line_id];
            if (line === undefined) {
                const message = `Cannot find line with id ${JSON.stringify(args.line_id)}: no such id in the list.`;
                return { error: "line_not_found", args: { message, fields: { line_id: args.line_id } } };
            }
            await line.loadState(args.path);
            return { command: "line_load", args: null };
        });
        server.bind("line_save", async args => {
            const line = this.activeLines[args.line_id];
            if (line === undefined) {
                const message = `Cannot find line with id ${JSON.stringify(args.line_id)}: no such id in the list.`;
                return { error: "line_not_found", args: { message, fields: { line_id: args.line_id } } };
            }
            const file = await line.saveState(args.path);
            return { command: "line_save", args: { path: file } };
        });
        server.bind("line_push", async args => {
            const line = this.activeLines[args.line_id];
            if (line === undefined) {
                const message = `Cannot find line with id ${JSON.stringify(args.line_id)}: no such id in the list.`;
                return { error: "line_not_found", args: { message, fields: { line_id: args.line_id } } };
            }
            await line.pushInput(args.tokens);
            return { command: "line_push", args: null };
        });
        server.bind("line_trim", async args => {
            const line = this.activeLines[args.line_id];
            if (line === undefined) {
                const message = `Cannot find line with id ${JSON.stringify(args.line_id)}: no such id in the list.`;
                return { error: "line_not_found", args: { message, fields: { line_id: args.line_id } } };
            }
            await line.trim(args.n_tokens);
            return { command: "line_trim", args: null };
        });
        server.bind("line_cancel", async args => {
            const line = this.activeLines[args.line_id];
            if (line === undefined) {
                const message = `Cannot find line with id ${JSON.stringify(args.line_id)}: no such id in the list.`;
                return { error: "line_not_found", args: { message, fields: { line_id: args.line_id } } };
            }
            await line.cancelInput();
            return { command: "line_cancel", args: null };
        });
        server.bind("line_start", async args => {
            const line = this.activeLines[args.line_id];
            if (line === undefined) {
                const message = `Cannot find line with id ${JSON.stringify(args.line_id)}: no such id in the list.`;
                return { error: "line_not_found", args: { message, fields: { line_id: args.line_id } } };
            }
            await line.start();
            return { command: "line_start", args: null };
        });
        server.bind("line_stop", async args => {
            const line = this.activeLines[args.line_id];
            if (line === undefined) {
                const message = `Cannot find line with id ${JSON.stringify(args.line_id)}: no such id in the list.`;
                return { error: "line_not_found", args: { message, fields: { line_id: args.line_id } } };
            }
            await line.stop();
            return { command: "line_stop", args: null };
        });
        server.bind("line_clear", async args => {
            const line = this.activeLines[args.line_id];
            if (line === undefined) {
                const message = `Cannot find line with id ${JSON.stringify(args.line_id)}: no such id in the list.`;
                return { error: "line_not_found", args: { message, fields: { line_id: args.line_id } } };
            }
            await line.clearState();
            await line.cancelInput();
            return { command: "line_clear", args: null };
        });
        server.bind("exit", async () => {
            await getModel().free();
            return { command: "exit", args: null };
        });
        server.bind("tokenize", async args => {
            const tokens = [...getModel().tokenize(args.text, args.parse_special ?? true, args.add_special ?? false)];
            return { command: "tokenize", args: { tokens } };
        });
        server.bind("detokenize", async args => {
            const text = getModel().detokenize(args.tokens, args.unparse_special ?? true, args.remove_special ?? false);
            return { command: "detokenize", args: { text } };
        });
    }
    public bind<Command extends keyof typeof SCommandArgs>(command: Command, cb: (args: z.output<typeof SCommandArgs[Command]>, socket: Socket) => PromiseOrNot<SResult | SError | SEvent | (SResult | SError | SEvent)[]>) {
        this.on("command", async (cmd, socket) => {
            if (cmd.command !== command) { return; }
            let to_send: (SError | SResult | SEvent)[];
            try {
                const result = await cb(cmd.args as any, socket);
                to_send = result instanceof Array ? result : [result];
            } catch (e) {
                let args: z.input<typeof SErrorArgs["internal_error"]>;
                if (typeof e === "object" && e !== null) {
                    args = { message: String(e), fields: Object.fromEntries(Object.entries(e).filter(([k, v]) => k !== "message")) };
                } else {
                    args = { message: String(e), fields: {} };
                }
                this.send(socket, { type: "error", query_id: cmd.query_id, object: { error: "internal_error", args } });
                return;
            }
            for (const elem of to_send) {
                if ("command" in elem) {
                    this.send(socket, { type: "result", query_id: cmd.query_id, object: elem });
                } else if ("error" in elem) {
                    this.send(socket, { type: "error", query_id: cmd.query_id, object: elem });
                } else if ("event" in elem) {
                    this.send(socket, { type: "event", object: elem });
                }
            }
        });
    }
    public log = (msg: string) => console.log(msg.trimEnd());
    public send(socket: Socket | null, message: SMessage) {
        if (message.type === "event") {
            this.log(`EVENT: ${JSON.stringify(message)}`);
            this.sockets.forEach(s => {
                s.write(`${JSON.stringify(message)}\n`);
            });
        } else {
            if (socket === null) { throw new Error(`cannot omit 'socket' argument when 'message' is not an event`); }
            if (message.type === "result") {
                this.log(`COMMAND RESULT (client ${this.socketIds.get(socket)}): ${JSON.stringify(message)}`);
            } else {
                this.log(`COMMAND ERROR (client ${this.socketIds.get(socket)}): ${JSON.stringify(message)}`);
            }
            socket.write(`${JSON.stringify(message)}\n`);
        }
    }
    public async listen(conn: ConnOption) {
        await new Promise((resolve, reject) => {
            this.server.on("listening", resolve);
            this.server.on("error", reject);
            if ("unix" in conn) {
                this.server.listen(conn.unix);
                this.log(`LISTENING at ${JSON.stringify("unix:" + conn.unix)}`);
            } else {
                this.server.listen(conn.port, conn.host ?? "localhost");
                this.log(`LISTENING at tcp:${conn.host ?? "localhost"}:${conn.port}`);
            }
        });
    }
    public modelResolve: null | (() => void) = null;
    public readonly close = createFreeEvent("close", () => {
        this.server.close();
        if (this.modelResolve !== null) {
            this.modelResolve();
            this.modelResolve = null;
        }
    });
    public async run(binariesPath: string) {
        await runModel({
            binaries_path: binariesPath,
            model_file: this.modelFile, log: this.log,
            callback: async model => {
                this.model = model;
                this.freeLines = [...model.lines];
                model.on("tokens", e => {
                    e.forEach(tokens => {
                        const line_id = Object.entries(this.activeLines).find(e => e[1] === tokens.line)?.[0];
                        if (line_id === undefined) { return; }
                        const next = tokens.next === null ? null : {
                            token: tokens.next.token,
                            piece: tokens.next.piece,
                            control: tokens.next.control,
                        };
                        const input = tokens.input.map(e => ({
                            token: e.token,
                            piece: e.piece,
                            control: e.control,
                        }))
                        const { entropy, replace, stop, stopReasons } = tokens;
                        this.send(null, {
                            type: "event",
                            object: {
                                event: "tokens",
                                args: { line_id, next, input, entropy, replace, stop, stopReasons }
                            }
                        });
                    });
                });
                model.on("llama_log", (level, message) => {
                    this.log(`LLAMA ${level.toUpperCase()} ${message.trim()}`);
                });
                model.on("generation_started", () => {
                    this.send(null, {
                        type: "event",
                        object: {
                            event: "generation_started",
                            args: null,
                        }
                    });
                });
                model.on("generation_stopped", () => {
                    this.send(null, {
                        type: "event",
                        object: {
                            event: "generation_stopped",
                            args: null,
                        }
                    });
                });
                await new Promise(async modelResolve => {
                    this.modelResolve = () => modelResolve(undefined);
                    process.on("exit", () => this.close());
                    model.on("model_free", () => this.close());
                    this.emit("model_loaded");
                });
            }
        });
    }
}

export const defaultTemplateString = `
{% for message in messages %}
# *{{ message['role'] }}*:
{{ message['content'] }}
{% endfor %}
`.trim();
export type ModelEventsRaw = {
    [k in keyof typeof SEventArgs]: [z.output<typeof SEventArgs[k]>]
};
export type ModelClientEvents = ModelEventsRaw & {
    message_json_error: [SyntaxError],
    message_schema_error: [z.ZodError<SMessage>],
    raw_message: [SMessage],
    client_close: [],
};
export class ModelClient extends EventEmitter<ModelClientEvents> {
    public static async create(params: {
        conn: { unix: string } | { host?: string, port: number },
        timeout?: number,
        fallbackStartServer?: undefined | {
            modelFile: string,
            modelParams: ModelParamsSerialized,
            stdout?: number | IOType | Stream | null,
            stderr?: number | IOType | Stream | null,
            timeout?: number
        }
    }) {
        const { conn, timeout: baseTimeout, fallbackStartServer } = params;
        let client: ModelClient;
        try {
            client = await ModelClient.connect(conn, baseTimeout ?? 500);
        } catch (e) {
            if (fallbackStartServer === undefined) {
                throw new Error(`server not available`);
            }
            const { modelFile, modelParams, stdout, stderr, timeout } = fallbackStartServer;
            const serverProc = fork(
                path.join(import.meta.dirname, "start-server.js"),
                [modelFile, JSON.stringify(conn), JSON.stringify(modelParams)],
                { detached: true, stdio: [null, stdout ?? null, stderr ?? "inherit", "ipc"] }
            );
            try {
                client = await ModelClient.connect(conn, Math.max(0, (timeout ?? 0) - (baseTimeout ?? 500)));
            } catch (e) {
                serverProc.kill("SIGKILL");
                throw Object.assign(new Error(`cannot start server in given timeout`), { reason: e });
            }
        }
        return client;
    }
    public static async connect(conn: ConnOption, timeout: number = 0): Promise<ModelClient> {
        if (timeout <= 0) {
            const params = ("unix" in conn ? { path: conn.unix } : { port: conn.port, host: conn.host ?? "localhost" }) as NetConnectOpts;
            const socket = await new Promise<Socket>((resolve, reject) => {
                const socket = createConnection(params, () => resolve(socket));
                socket.on("error", reject);
            });
            const client = new ModelClient(socket, {}, {}, new Template(defaultTemplateString));
            const { metadata, model_params } = await client.exec("start", null);
            Object.freeze(Object.assign(client.modelMetadata, metadata));
            Object.freeze(Object.assign(client.modelParams, model_params));
            const tempalteStr = metadata["tokenizer.chat_template"];
            if (tempalteStr === undefined) {
                throw new Error(`cannot extract model's template from metadata`);
            }
            client.template.parsed = new Template(tempalteStr).parsed;
            return client;
        } else {
            const errTimeout = new Error(`connection timed out`);
            const end = Date.now() + timeout;
            while (true) {
                try {
                    return await new Promise(async (resolve, reject) => {
                        const delay = end - Date.now();
                        if (delay < 0) {
                            reject(errTimeout);
                            return;
                        }
                        setTimeout(() => reject(errTimeout), delay);
                        try {
                            resolve(await ModelClient.connect(conn));
                        } catch (e) {
                            reject(e);
                        }
                    });
                } catch (e) {
                    const err = e as any;
                    if (err.syscall === "connect") {
                        await new Promise(resolve => setTimeout(resolve, 50));
                        continue;
                    }
                    throw err;
                }
            }
        }
    }
    public buffer = "";
    public constructor(
        public readonly socket: Socket,
        public readonly modelMetadata: Record<string, string>,
        public readonly modelParams: ModelParamsSerialized,
        public readonly template: Template,
    ) {
        super();
        this.setMaxListeners(50);
        this.socket.on("data", piece => {
            this.buffer += piece.toString();
            const parts = this.buffer.split("\n");
            if (parts.length > 1) {
                this.buffer = parts.pop() as string;
                parts.forEach(part => {
                    let raw: object;
                    try {
                        raw = JSON.parse(part);
                    } catch (e) {
                        console.error("Message JSON error:", e);
                        this.emit("message_json_error", e as SyntaxError);
                        return;
                    }
                    const parsed = z.safeParse(SMessageSchema, raw);
                    if (!parsed.success) {
                        console.error("Message schema error:", parsed.error.issues);
                        this.emit("message_schema_error", parsed.error);
                        return;
                    }
                    const msg = parsed.data;
                    this.emit("raw_message", msg);
                    if (msg.type === "event") {
                        this.emit(msg.object.event, msg.object.args as any);
                    }
                });
            }
        });
        this.on("command_json_error", ({ message }) => console.error(message));
        this.on("command_schema_error", ({ issues }) => console.error("Server Schema Error", issues));
        this.prefixes = {
            initToSystem: this.scheme({ messages: [{ role: "system", content: "\uE001" }] }),
            initToUser: this.scheme({ messages: [{ role: "user", content: "\uE001" }] }),
            systemToUser: this.scheme({ messages: [{ role: "system", content: "\uE000" }, { role: "user", content: "\uE001" }] }),
            userToAssistant: this.scheme({ messages: [{ role: "user", content: "\uE000" }, { role: "assistant", content: "\uE001" }] }),
            assistantToUser: this.scheme({ messages: [{ role: "user", content: "..." }, { role: "assistant", content: "\uE000" }, { role: "user", content: "\uE001" }] }),
            toolToAssistant: this.scheme({ messages: [{ role: "user", content: "..." }, { role: "assistant", content: "..." }, { role: "tool", content: "\uE000" }, { role: "assistant", content: "\uE001" }] }),
            assistantToTool: this.scheme({ messages: [{ role: "user", content: "..." }, { role: "assistant", content: "\uE000" }, { role: "tool", content: "\uE001" }] }),
        };
    }
    public readonly prefixes: {
        initToSystem: ContentElem,
        initToUser: ContentElem,
        systemToUser: ContentElem,
        userToAssistant: ContentElem,
        assistantToUser: ContentElem,
        toolToAssistant: ContentElem,
        assistantToTool: ContentElem,
    };
    public static rng = new Yurandom(`${process.pid}_${Date.now()}`);
    public async exec<Command extends keyof typeof SCommandArgs>(command: Command, args: z.output<typeof SCommandArgs[Command]>) {
        const query_id = ModelClient.rng.uuid();
        const pkg = { command, query_id, args } as SCommand;
        return await new Promise<z.output<typeof SResultArgs[Command]>>((resolve, reject) => {
            this.on("raw_message", msg => {
                if ("query_id" in msg && msg.query_id === query_id) {
                    if (msg.type === "error") {
                        reject(Object.assign(new Error(msg.object.error), msg.object.args));
                    } else {
                        resolve(msg.object.args as any);
                    }
                }
            });
            this.socket.write(JSON.stringify(pkg) + "\n");
        });
    }
    public scheme(constructor: TemplateInput, startKey: string = "\uE000", endKey: string = "\uE001") {
        const text = this.template.render(constructor);
        const content = extractMiddle(text, startKey, endKey);
        if (content === undefined) {
            throw new Error(`cannot extract scheme from given pattern`);
        }
        return { special: true, text: content };
    }

    public async closeServer() {
        await this.exec("exit", null);
        await this.close();
    }
    public readonly close = createFreeEvent<ModelClient, Promise<void>>("client_close", async () => {
        await new Promise(resolve => this.socket.end(() => resolve(undefined)));
    });
}



export function packTokens(tokens: Token[]) {
    const content: (number | string)[] = [];
    tokens.forEach(e => {
        if (e.control) {
            content.push(e.token);
        } else {
            if (typeof content.at(-1) === "string") {
                content.push((content.pop() as string) + e.piece);
            } else {
                content.push(e.piece);
            }
        }
    });
    const text = content.findLast(e => typeof e === "string");
    return { content, text };
}
export type Token = z.output<typeof SToken>;
export type TokensEvent = z.output<typeof SEventArgs["tokens"]>;
export type TokenSequence = {
    tokens: Token[],
    entropy: number,
    next: Token | null,
    stopReasons: StopReason[]
};
export type StopCondition = InferenceLineParams;
export type ContentElem = string | number | number[] | { special: boolean, text: string };
export type TemplateInput = {
    messages: {
        role: "system" | "user" | "assistant" | "tool",
        content: Serializable,
        [k: string | number]: Serializable,
    }[],
    [k: string | number]: Serializable,
};


export class ClientLine {
    public async loadContent(file: string) {
        await this.cancel();
        this.tokens = []
        await this.pullRaw(() => this.client.exec("line_load", { line_id: this.lineId, path: file }));
    }
    public async saveContent(file: string) {
        await this.client.exec("line_save", { line_id: this.lineId, path: file });
    }
    public sampler: SamplerConstructor = [{ type: "greedy" }];
    public constructor(public readonly client: ModelClient, public readonly lineId: string) { }
    public static async create(client: ModelClient, lineId?: string, sampler: SamplerConstructor = [{ type: "greedy" }]) {
        lineId = (await client.exec("line_init", { line_id: lineId, sampler, inference: { eog_stop: false } })).line_id;
        await client.exec("line_stop", { line_id: lineId });
        await client.exec("line_clear", { line_id: lineId });
        await client.exec("line_cancel", { line_id: lineId });
        return new ClientLine(client, lineId);
    }
    public async setSampler(sampler: SamplerConstructor) {
        await this.client.exec("line_init", { line_id: this.lineId, sampler });
        this.sampler = sampler;
    }
    public tokens: Token[] = [];
    public unparsedTokens: number[] = [];
    public async pullRaw(action: () => void, stopCond: (events: TokensEvent[]) => boolean = e => !!e.at(-1)?.stop) {
        const { last, replace, tokens } = await new Promise<{ tokens: Token[], replace: boolean, last: TokensEvent }>(async resolve => {
            let tokens: Token[] = [];
            let replace = false;
            let events: TokensEvent[] = [];
            const handler = (e: TokensEvent) => {
                if (e.line_id !== this.lineId) { return; }
                events.push(e);
                if (e.replace) {
                    if (tokens.length) {
                        tokens.pop();
                    } else {
                        replace = true;
                    }
                }
                tokens.push(...e.input);
                if (stopCond(events)) {
                    this.client.off("tokens", handler);
                    resolve({ tokens, replace, last: e });
                }
            };
            this.client.on("tokens", handler);
            action();
        });
        const { entropy, next, stopReasons } = last;
        if (replace) {
            this.tokens.pop();
        }
        this.tokens.push(...tokens);
        this.unparsedTokens = next === null ? [] : [next.token];
        return { tokens, entropy, replace, next, stopReasons } as TokenSequence;
    }
    public async pull(stop: StopCondition) {
        const prefixSize = this.unparsedTokens.length;
        await this.client.exec("line_init", { line_id: this.lineId, inference: stop });
        const { tokens, entropy, next, stopReasons } = await this.pullRaw(() => this.client.exec("line_start", { line_id: this.lineId }));
        const { content, text } = packTokens(tokens.slice(prefixSize));
        return { content, tokens: tokens.slice(prefixSize), text, entropy, next, stopReasons };
    }
    public async push(...content: ContentElem[]) {
        const clearContent: { special: boolean, chunk: string | number[] }[] = [];
        function parse(e: ContentElem) {
            if (typeof e === "string") { return { special: false, chunk: e }; }
            if (e instanceof Array) { return { special: true, chunk: [...e] }; }
            if (typeof e === "number") { return { special: true, chunk: [e] }; }
            return { special: e.special, chunk: e.text };
        }
        content.forEach(e => {
            const prev = clearContent.at(-1);
            if (typeof prev?.chunk === "string") {
                clearContent.pop();
                const current = parse(e);
                if (prev.special === current.special) {
                    clearContent.push({ special: prev.special, chunk: prev.chunk + current.chunk });
                } else {
                    clearContent.push(prev);
                    clearContent.push(current);
                }
            } else {
                clearContent.push(parse(e));
            }
        });
        const tokens = (await Promise.all(clearContent.map(async e => {
            if (typeof e.chunk === "string") {
                return (await this.client.exec("tokenize", { text: e.chunk, parse_special: e.special })).tokens;
            } else {
                return e.chunk;
            }
        }))).flat();
        await this.client.exec('line_push', { line_id: this.lineId, tokens });
        this.unparsedTokens.push(...tokens);
        return tokens.length;
    }
    public async cancel() {
        this.unparsedTokens = [];
        await this.client.exec("line_cancel", { line_id: this.lineId });
    }
    public async trim(nTokens: number, cancel: boolean = true) {
        this.tokens = this.tokens.slice(0, this.tokens.length - nTokens);
        await this.client.exec("line_trim", { line_id: this.lineId, n_tokens: nTokens });
        await this.client.exec("line_cancel", { line_id: this.lineId });
        if (cancel) {
            this.unparsedTokens = [];
        } else {
            await this.client.exec("line_push", { line_id: this.lineId, tokens: this.unparsedTokens });
        }
    }
    public async clear() {
        await this.client.exec("line_clear", { line_id: this.lineId });
        await this.cancel();
        this.tokens = [];
        this.unparsedTokens = [];
    }
    public async free() {
        await this.client.exec("line_free", { line_id: this.lineId });
    }
}








//
