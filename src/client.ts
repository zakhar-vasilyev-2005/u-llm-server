import path from "path";
import { ModelParamsSchema, type ModelParamsSerialized } from "./llama-base.js";
import { SResultArgs, SCommandArgs, SEventArgs, SMessageSchema, type SCommand, type SMessage, SToken } from './server-schemas.js';
import { createConnection, Socket, type NetConnectOpts } from "net";
import { EventEmitter } from "events";
import * as z from "zod";
import { Yurandom } from 'yurandom';
import { createFreeEvent } from "./event-util.js";
import { extractMiddle } from "./extract-middle.js";
import { Template } from "@huggingface/jinja";
import { ChildProcess, fork } from "child_process";
import type { Serializable } from "./serializable.js";
import { GrowBuffer } from "./growbuffer.js";
import type { InputElem } from "./model.js";
import { ModelVocab } from "./vocab.js";
import { blendObjects } from "./typeutils.js";
import { getPathToEmbeddedBinaries, getPathToLlama } from "./embedded_binaries_path.js";




export const defaultTemplateString = `
{{- bos_token }}
{% for message in messages %}<|start_message|>{{ message['role'] }}

{{ message['content'] }}
<|end_of_message|>{% endfor %}
`.trim();
export type ModelEventsRaw = {
    [k in keyof typeof SEventArgs]: [z.output<typeof SEventArgs[k]>]
};
export type ModelClientEvents = ModelEventsRaw & {
    socket_error: [Error],
    message_json_error: [SyntaxError],
    message_schema_error: [z.ZodError<SMessage>],
    raw_message: [SMessage],
    client_close: [],
};
export const ModelClientParamsScheme = z.object({
    conn: z.union([z.object({
        unix: z.string(),
        host: z.never().optional(),
        port: z.never().optional(),
        timeout: z.number().nonnegative().optional(),
    }), z.object({
        unix: z.never().optional(),
        host: z.string().optional(),
        port: z.int().min(1024).max(65535),
        timeout: z.number().nonnegative().optional(),
    })]),
    vocabFile: z.string(),
    fallbackStartServer: z.object({
        modelFile: z.string(),
        modelParams: ModelParamsSchema,
        stdout: z.union([z.int().positive(), z.enum(["ignore", "inherit"])]).optional(),
        stderr: z.union([z.int().positive(), z.enum(["ignore", "inherit"])]).optional(),
        timeout: z.number().nonnegative().optional(),
    }).optional(),
});
export type ModelClientParams = z.output<typeof ModelClientParamsScheme>;
export type TemplateInput = {
    messages: {
        role: ChatRole,
        content: Serializable,
        [k: string | number]: Serializable,
    }[],
    [k: string | number]: Serializable,
};
export type Token = z.output<typeof SToken>;
export type ContentElemBase = string | number | Int32Array | Token | InputElem | TemplateInput;
export type ContentElem = ContentElemBase | ContentElemBase[];
export type ChatRole = "system" | "user" | "assistant" | "tool";
export class ModelClient extends EventEmitter<ModelClientEvents> {
    public static async create(params: ModelClientParams) {
        let { conn, fallbackStartServer } = params;;
        conn.timeout ??= 500;
        let client: ModelClient;
        try {
            client = await ModelClient.connect(params, conn.timeout);
        } catch (e) {
            if (fallbackStartServer === undefined) {
                throw new Error(`server not available`);
            }
            let { modelFile, modelParams, stdout, stderr, timeout: startTimeout } = fallbackStartServer;
            stderr ??= "inherit";
            stdout ??= "ignore";
            startTimeout ??= 0;
            const serverProc = fork(
                path.join(import.meta.dirname, "start-server.js"),
                [modelFile, JSON.stringify(conn), JSON.stringify(modelParams)],
                { detached: true, stdio: [null, stdout, stderr, "ipc"] }
            );
            try {
                client = await ModelClient.connect(params, startTimeout, serverProc);
            } catch (e) {
                serverProc.kill("SIGKILL");
                throw Object.assign(new Error(`cannot start server in given timeout (timeout = ${startTimeout}ms)`), { reason: e });
            }
        }
        return client;
    }
    public static async connect(params: ModelClientParams, timeout: number, serverProc: ChildProcess | undefined = undefined): Promise<ModelClient> {
        const { conn } = params;
        if (timeout <= 0) {
            const connParams = (conn.unix !== undefined ? { path: conn.unix } : { port: conn.port, host: conn.host ?? "localhost" }) as NetConnectOpts;
            let errBuffer: Error[] = [];
            const errBufferizer = (err: Error) => errBuffer.push(err);
            const socket = await new Promise<Socket>((resolve, reject) => {
                const socket = createConnection(connParams, () => resolve(socket));
                socket.on("error", errBufferizer);
                socket.once("error", reject);
            });
            const vocab = new ModelVocab(getPathToLlama(getPathToEmbeddedBinaries()), params.vocabFile);
            const client = new ModelClient(socket, {}, {}, new Template(defaultTemplateString), vocab, serverProc);
            const errRouter = (err: Error) => client.emit("socket_error", err);
            socket.off("error", errBufferizer);
            errBuffer.forEach(errRouter);
            socket.on("error", errRouter);
            const { metadata, model_params } = await client.exec("start", null);
            Object.assign(client.modelMetadata, metadata);
            Object.assign(client.modelParams, model_params);
            const tempalteStr = metadata["tokenizer.chat_template"];
            if (tempalteStr === undefined) {
                throw new Error(`cannot extract model's template from metadata`);
            }
            Object.assign(client.template, new Template(tempalteStr));
            Object.assign(client.prefixes, client.createPrefixes());
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
                            resolve(await ModelClient.connect(params, 0));
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
    protected constructor(
        public readonly socket: Socket,
        public readonly modelMetadata: Record<string, string>,
        public readonly modelParams: ModelParamsSerialized,
        public readonly template: Template,
        public readonly vocab: ModelVocab,
        public readonly serverProc: ChildProcess | undefined,
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
        this.prefixes = this.createPrefixes();
    }
    public createPrefixes() {
        return {
            initToSystem: this.scheme({ messages: [{ role: "system", content: "\uE001" }] }),
            initToUser: this.scheme({ messages: [{ role: "user", content: "\uE001" }] }),
            systemToUser: this.scheme({ messages: [{ role: "system", content: "\uE000" }, { role: "user", content: "\uE001" }] }),
            userToAssistant: this.scheme({ messages: [{ role: "user", content: "\uE000" }, { role: "assistant", content: "\uE001" }] }),
            assistantToUser: this.scheme({ messages: [{ role: "user", content: "..." }, { role: "assistant", content: "\uE000" }, { role: "user", content: "\uE001" }] }),
            toolToAssistant: this.scheme({ messages: [{ role: "user", content: "..." }, { role: "assistant", content: "..." }, { role: "tool", content: "\uE000" }, { role: "assistant", content: "\uE001" }] }),
            assistantToTool: this.scheme({ messages: [{ role: "user", content: "..." }, { role: "assistant", content: "\uE000" }, { role: "tool", content: "\uE001" }] }),
            userToGeneration: this.scheme({ messages: [{ role: "user", content: "...\uE000" }], add_generation_prompt: true }),
        };
    }
    public readonly prefixes: ReturnType<ModelClient["createPrefixes"]>;
    public static rng = new Yurandom(`${process.pid}_${Date.now()}`);
    public async exec<Command extends keyof typeof SCommandArgs>(command: Command, args: z.output<typeof SCommandArgs[Command]>) {
        const query_id = ModelClient.rng.uuid();
        const pkg = { command, query_id, args } as SCommand;
        return await new Promise<z.output<typeof SResultArgs[Command]>>((resolve, reject) => {
            const handler = (msg: SMessage) => {
                if ("query_id" in msg && msg.query_id === query_id) {
                    this.off("raw_message", handler);
                    if (msg.type === "error") {
                        reject(Object.assign(new Error(msg.object.error), msg.object.args));
                    } else {
                        resolve(msg.object.args as any);
                    }
                }
            };
            this.on("raw_message", handler);
            this.socket.write(JSON.stringify(pkg) + "\n");
        });
    }
    public scheme(constructor: TemplateInput, startKey: string = "\uE000", endKey: string = "\uE001") {
        const prefix = {
            bos_token: this.vocab.special.add_bos ? this.vocab.special.bos?.piece ?? "" : "",
            eos_token: this.vocab.special.add_eos ? this.vocab.special.eos?.piece ?? "" : "",
        };
        const suffix = {
            add_generation_prompt: !!constructor["add_generation_prompt"],
        };
        const text = this.template.render(blendObjects(prefix, constructor, suffix));
        const content = extractMiddle(text, startKey, endKey);
        if (content === undefined) {
            throw new Error(`cannot extract scheme from given pattern`);
        }
        return { special: true, text: content };
    }
    public parse(...content: ContentElem[]) {
        const elems = (content
            .flatMap(e => e instanceof Array ? e : [e])
            .map(e => {
                if (typeof e === "string") {
                    return { special: false, text: e };
                } else if (typeof e === "number") {
                    return { tokens: [e] };
                } else if ("messages" in e) {
                    return this.scheme(e);
                } else if ("piece" in e) {
                    return { special: e.special, text: e.piece };
                } else if (e instanceof Int32Array) {
                    return { tokens: [...e] };
                } else {
                    return e;
                }
            })
        ) as InputElem[];
        const input: InputElem[] = [];
        for (const elem of elems) {
            const last = input.at(-1);
            if (last?.tokens !== undefined && elem.tokens !== undefined) {
                input.pop();
                last.tokens.push(...elem.tokens);
                input.push(last);
            } else {
                input.push(elem);
            }
        }
        const pieces = input.flatMap(elem => {
            if (elem.tokens === undefined) {
                return [elem];
            } else {
                let pieces: { special: boolean, text: string }[] = [];
                let buf = new GrowBuffer(100);
                let special = false
                const flush = () => {
                    pieces.push({ special, text: buf.buffer.toString("utf8") });
                    buf.clear(100);
                };
                for (const token of elem.tokens) {
                    if (this.vocab.isSpecial(token) !== special) {
                        flush();
                        special = this.vocab.isSpecial(token);
                    }
                    buf.push(this.vocab.toPiece(token));
                }
                flush();
                return pieces;
            }
        });
        const result: typeof pieces = [];
        for (const piece of pieces) {
            const last = result.at(-1);
            if (last?.special === piece.special) {
                result.pop();
                last.text += piece.text;
                result.push(last);
            } else {
                result.push(piece);
            }
        }
        return result;
    }
    public async closeServer() {
        await this.exec("exit", null);
        await this.close();
    }
    public readonly close = createFreeEvent<ModelClient, Promise<void>>("client_close", async () => {
        await new Promise(resolve => this.socket.end(() => resolve(undefined)));
    });
}










//