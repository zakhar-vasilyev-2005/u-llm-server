import path from "path";
import { type ModelParamsSerialized, type SamplerConstructor } from "./llama-base.js";
import { SResultArgs, SCommandArgs, SEventArgs, SMessageSchema, type SCommand, type SMessage, SToken } from './server-schemas.js';
import { createConnection, Socket, type NetConnectOpts } from "net";
import { EventEmitter } from "events";
import * as z from "zod";
import type { Serializable } from "./serializable.js";
import { Yurandom } from 'yurandom';
import { createFreeEvent } from "./event-util.js";
import { extractMiddle } from "./extract-middle.js";
import type { InferenceLineParams, InputElem, StopReason } from "./model.js";
import { Template } from "@huggingface/jinja";
import { fork, type IOType } from "child_process";
import type Stream from "stream";
import type { ConnOption } from "./server.js";

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
    message_json_error: [SyntaxError],
    message_schema_error: [z.ZodError<SMessage>],
    raw_message: [SMessage],
    client_close: [],
};
export interface ModelClientParams {
    conn: { unix: string } | { host?: string, port: number },
    timeout?: number,
    fallbackStartServer?: undefined | {
        modelFile: string,
        modelParams: ModelParamsSerialized,
        stdout?: number | IOType | Stream | null,
        stderr?: number | IOType | Stream | null,
        timeout?: number
    }
};
export class ModelClient extends EventEmitter<ModelClientEvents> {
    public static async create(params: ModelClientParams) {
        let { conn, timeout: connectTimeout, fallbackStartServer } = params;;
        connectTimeout ??= 500;
        let client: ModelClient;
        try {
            client = await ModelClient.connect(conn, connectTimeout);
        } catch (e) {
            if (fallbackStartServer === undefined) {
                throw new Error(`server not available`);
            }
            let { modelFile, modelParams, stdout, stderr, timeout: startTimeout } = fallbackStartServer;
            stderr ??= "inherit";
            stdout ??= null;
            startTimeout ??= 0;
            const serverProc = fork(
                path.join(import.meta.dirname, "start-server.js"),
                [modelFile, JSON.stringify(conn), JSON.stringify(modelParams)],
                { detached: true, stdio: [null, stdout, stderr, "ipc"] }
            );
            try {
                client = await ModelClient.connect(conn, Math.max(0, startTimeout));
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
            Object.assign(client.template, new Template(tempalteStr));
            Object.freeze(Object.assign(client.prefixes, client.createPrefixes()));
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
    protected constructor(
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



export type PackedTokens = {
    content: (string | number)[],
    text?: string | undefined
};
export function packTokens(tokens: Token[], prevState: PackedTokens = { content: [] }): PackedTokens {
    const content: (number | string)[] = prevState.content;
    tokens.forEach(e => {
        if (e.special) {
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
export type StopPredicateArg = Omit<PullResult, "entropy"> & {
    lastToken: Token,
    entropy: number | null,
    stop: boolean,
};
export type StopCondition = InferenceLineParams & {
    stop_predicate?: (data: StopPredicateArg) => boolean
};
export type ContentElem = string | number | number[] | InputElem | TemplateInput;
export type ChatRole = "system" | "user" | "assistant" | "tool";
export type TemplateInput = {
    messages: {
        role: ChatRole,
        content: Serializable,
        [k: string | number]: Serializable,
    }[],
    [k: string | number]: Serializable,
};


export type PullResult = PackedTokens & TokenSequence;
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
    public async setSampler(sampler: SamplerConstructor, samplerOffset: number = 0) {
        await this.client.exec("line_init", { line_id: this.lineId, sampler, sampler_offset: samplerOffset });
        this.sampler = sampler;
    }
    public tokens: Token[] = [];
    public unparsedContent: InputElem[] = [];
    public async pullRaw(action: () => void, stopCond: (events: TokensEvent[]) => boolean = e => !!e.at(-1)?.stop) {
        const { last, tokens } = await new Promise<{ tokens: Token[], last: TokensEvent }>(async resolve => {
            let tokens: Token[] = [];
            let events: TokensEvent[] = [];
            const handler = (e: TokensEvent) => {
                if (e.line_id !== this.lineId) { return; }
                events.push(e);
                tokens.push(...e.input);
                if (stopCond(events)) {
                    this.client.off("tokens", handler);
                    resolve({ tokens, last: e });
                }
            };
            this.client.on("tokens", handler);
            action();
        });
        const { entropy, next, stopReasons } = last;
        this.tokens.push(...tokens);
        this.unparsedContent = next === null ? [] : [{ tokens: [next.token] }];
        return { tokens, entropy, next, stopReasons } as TokenSequence;
    }
    public async pull(stop: StopCondition) {
        await this.client.exec("line_init", { line_id: this.lineId, inference: stop });
        const { stop_predicate } = stop;
        let packed: PackedTokens = { content: [] };
        let packedTokens: Token[] = [];
        const startPos = this.tokens.length;
        const { tokens, entropy, next, stopReasons } = await this.pullRaw(
            () => this.client.exec("line_start", { line_id: this.lineId }),
            stop_predicate === undefined ? undefined : (events => {
                const last = events.at(-1);
                const eventsTokens = events.flatMap(e => e.input);
                packed = packTokens(eventsTokens.slice(packedTokens.length), packed);
                packedTokens = eventsTokens;
                return (
                    last === undefined ||
                    stop_predicate({
                        lastToken: eventsTokens.at(-1) as Token,
                        tokens: eventsTokens,
                        content: packed.content,
                        text: packed.text,
                        entropy: last.entropy,
                        next: last.next,
                        stopReasons: last.stopReasons,
                        stop: last.stop,
                    }) ||
                    last.stop
                );
            })
        );
        await this.client.exec("line_stop", { line_id: this.lineId });
        await this.goto(startPos + packedTokens.length);
        if (next !== null) {
            this.unparsedContent.push({ text: next.piece, special: next.special });
        }
        return { content: packed.content, tokens, text: packed.text, entropy, next, stopReasons } as PullResult; 1.
    }
    public static parseContent(e: ContentElem): InputElem {
        if (typeof e === "string") { return { special: false, text: e }; }
        if (e instanceof Array) { return { tokens: [...e] }; }
        if (typeof e === "number") { return { tokens: [e] }; }
        return { special: e.special, text: e.text };
    }
    public async reset(...content: ContentElem[]) {
        const pieces = await Promise.all(content.map(ClientLine.parseContent).map(e => e?.tokens !== undefined
            ? this.client.exec("detokenize", { tokens: e.tokens, unparse_special: true }).then(e => ({ text: e.text, special: true }))
            : e
        )) as { text: string, special: boolean }[];
        let maxTokens: number = 0;
        for (const token of this.tokens) {
            const piece = pieces.shift();
            if (piece === undefined) { break; }
            if (token.special && !piece.special) { break; }
            if (piece.text.startsWith(token.piece)) {
                const newPiece = { text: piece.text.slice(token.piece.length), special: piece.special };
                if (newPiece.text.length !== 0) {
                    pieces.unshift(newPiece);
                }
            } else {
                break;
            }
            maxTokens++;
        }
        await this.trim(this.tokens.length - maxTokens);
        await this.push(...pieces);
    }
    public async push(...content: ContentElem[]) {
        const elems = content.map(ClientLine.parseContent);
        await this.client.exec('line_push', { line_id: this.lineId, content: elems });
        this.unparsedContent.push(...elems);
    }
    public async step(...content: ContentElem[]) {
        await this.push(...content);
        await this.pull({ max_tokens: 0 });
        await this.cancel();
        return this.tokens.length;
    }
    public async cancel() {
        this.unparsedContent = [];
        await this.client.exec("line_cancel", { line_id: this.lineId });
    }
    public async trim(nTokens: number) {
        if (nTokens < 0) {
            throw new Error(`nTokens expected to be >=0`);
        }
        if (nTokens === 0) { return; }
        this.tokens = this.tokens.slice(0, this.tokens.length - nTokens);
        await this.client.exec("line_trim", { line_id: this.lineId, n_tokens: nTokens });
        await this.cancel();
    }
    public async trimText(nChars: number) {
        await this.cancel();
        let nTokens: number = 0;
        for (const token of this.tokens.reverse()) {
            if (nChars > token.piece.length) {
                nChars -= token.piece.length;
            } else {
                const piece = token.piece.slice(token.piece.length - nChars);
                await this.push({ text: piece, special: false });
                nChars = 0;
            }
            nTokens++;
            if (nChars === 0) { break; }
        }
        await this.trim(nTokens);
    }
    public async goto(nTokens: number) {
        if (nTokens > this.tokens.length) {
            throw new Error(`cannot return to given nTokens: there's less tokens generated than the given nTokens value`);
        }
        await this.trim(this.tokens.length - nTokens);
    }
    public async clear() {
        await this.client.exec("line_clear", { line_id: this.lineId });
        this.tokens = [];
        this.unparsedContent = [];
    }
    public async free() {
        await this.client.exec("line_free", { line_id: this.lineId });
    }
}







//