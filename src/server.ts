import path from "path";
import { Backend, ModelLine, Model } from "./llama.js";
import { type ContextParams, type ModelParamsSerialized } from "./llama-base.js";
import { SCommandArgs, SCommandSchema, SErrorArgs, type SCommand, type SError, type SEvent, type SMessage, type SResult } from './server-schemas.js';
import { createServer, Server, Socket } from "net";
import { EventEmitter } from "events";
import * as z from "zod";
import type { Serializable } from "./serializable.js";
import { sprintf } from "sprintf-js";
import { Yurandom } from 'yurandom';
import { createFreeEvent } from "./event-util.js";



type PromiseOrNot<T> = T | Promise<T>;
export type ConnOption = { unix: string, host?: undefined, port?: undefined } | { unix?: undefined, host?: string | undefined, port: number };


export const runModelDefaults = {
    modelParams: {
        n_gpu_layers: 999,
        check_tensors: false,
        use_extra_bufts: true,
    } as ModelParamsSerialized,
    contextParams: {
        flash_attn_type: "enabled",
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
    model_params: ModelParamsSerialized,
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
        const model_params = Object.assign(params.model_params, {
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
                socket.on("error", err => {
                    this.log(`SOCKET ERROR (client ${this.socketIds.get(socket)}): ${err.message} (fields: ${JSON.stringify(Object.assign({}, err), undefined, 4)})`);
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
            let line: ModelLine | undefined = this.activeLines[line_id];
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
                await line.setSampler(args.sampler, args.sampler_offset ?? 0);
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
            await line.pushInput(args.content);
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
            if (conn.unix !== undefined) {
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
            model_params: this.modelParams,
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
                            special: tokens.next.special,
                        };
                        const input = tokens.input.map(e => ({
                            token: e.token,
                            piece: e.piece,
                            special: e.special,
                        }))
                        const { entropy, stop, stopReasons } = tokens;
                        this.send(null, {
                            type: "event",
                            object: {
                                event: "tokens",
                                args: { line_id, next, input, entropy, stop, stopReasons }
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








//
