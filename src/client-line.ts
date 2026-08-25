import { type SamplerConstructor } from "./llama-base.js";
import { SEventArgs, SToken } from './server-schemas.js';
import * as z from "zod";
import type { InferenceLineParams, InputElem, StopReason } from "./model.js";
import type { ModelClient, TemplateInput } from "./client.js";
import { stripField } from "./typeutils.js";




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
export type StopPredicateArg = {
    next: Token | null;
    entropy: number | null;
    stopReasons: StopReason[];
    stop: boolean;
    text?: string | undefined;
    textSpecial: string,
    tokensRecieved: Token[];
    tokensRecievedNow: Token[],
    content: (string | number)[];
};
export type StopCondition = InferenceLineParams & {
    stop_predicate?: ((data: StopPredicateArg) => boolean) | undefined,
};
export type ContentElem = string | number | number[] | InputElem | TemplateInput;


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
    public async pull(stop: Q | StopCondition) {
        let stopCondition: StopCondition;
        if (stop instanceof Q) {
            const q = stop;
            stopCondition = Object.assign({ stop_predicate: data => q.stopCondition(data) } as StopCondition, q.inferenceParams ?? {});
        } else {
            stopCondition = stop;
        }
        const { stop_predicate } = stopCondition;
        await this.client.exec("line_init", { line_id: this.lineId, inference: stripField(stopCondition, "stop_predicate") });
        let packed: PackedTokens = { content: [] };
        let packedTokens: Token[] = [];
        let endPos: number | undefined = undefined;
        let textFull = "";
        const { tokens, entropy, next, stopReasons } = await this.pullRaw(
            () => this.client.exec("line_start", { line_id: this.lineId }),
            stop_predicate === undefined ? undefined : (events => {
                const last = events.at(-1);
                const eventsTokens = events.flatMap(e => e.input);
                const unpackedTokens = eventsTokens.slice(packedTokens.length);
                textFull += unpackedTokens.map(e => e.piece).join("");
                packed = packTokens(unpackedTokens, packed);
                packedTokens = eventsTokens;
                if (last === undefined || last.stop || stop_predicate({
                    tokensRecieved: eventsTokens,
                    tokensRecievedNow: unpackedTokens,
                    content: packed.content,
                    text: packed.text,
                    textSpecial: textFull,
                    entropy: last.entropy,
                    next: last.next,
                    stopReasons: last.stopReasons,
                    stop: last.stop,
                })) {
                    endPos = eventsTokens.length;
                    this.client.exec("line_stop", { line_id: this.lineId }).catch(e => { throw e; });
                    return true;
                }
                return false;
            })
        );
        if (endPos !== undefined) {
            await this.goto(endPos);
        }
        if (next !== null) {
            this.unparsedContent.push({ text: next.piece, special: next.special });
        }
        await this.cancel();
        return { content: packed.content, tokens, text: packed.text, entropy, next, stopReasons } as PullResult; 1.
    }
    public static parseContent(e: ContentElem): InputElem {
        if (typeof e === "string") { return { special: false, text: e }; }
        if (e instanceof Array) { return { tokens: [...e] }; }
        if (typeof e === "number") { return { tokens: [e] }; }
        return { special: e.special, text: e.text };
    }
    public async resetConfig(...content: ContentElem[]) {
        const pieces = await Promise.all(content.map(ClientLine.parseContent).map(e => e?.tokens !== undefined
            ? this.client.exec("detokenize", { tokens: e.tokens, unparse_special: true }).then(e => ({ text: e.text, special: true }))
            : e
        )) as { text: string, special: boolean }[];
        let nKeep: number = 0;
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
            nKeep++;
        }
        return { nKeep, nTrim: this.tokens.length - nKeep, input: pieces };
    }
    public async reset(...content: ContentElem[]) {
        const { nKeep, input } = await this.resetConfig(...content);
        await this.goto(nKeep);
        await this.push(...input);
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

export type QRegexOptions = {
    special_token?: "restart_regex" | "end_regex" | "include" | undefined,
};
export type QSubstringOptions = QRegexOptions & {
    ignore_case?: boolean | undefined,
};
export type QOnMatch = (data: StopPredicateArg, m: RegExpExecArray, source: { tokens: Token[] } & PackedTokens) => Q | undefined | boolean;
export class Q {
    public constructor(
        public readonly stopCondition: StopCondition["stop_predicate"] extends (infer T) | undefined ? T : never,
        public readonly inferenceParams?: InferenceLineParams | undefined,
    ) { }
    public static some(...conditions: Q[]) {
        return new Q(data => conditions.some(e => e.stopCondition(data)));
    }
    public static every(...conditions: Q[]) {
        return new Q(data => conditions.every(e => e.stopCondition(data)));
    }
    public not() {
        return new Q(data => !this.stopCondition(data));
    }
    public with(inferenceParams: InferenceLineParams) {
        return new Q(this.stopCondition, inferenceParams);
    }
    public static regex(pattern: RegExp, onMatch: QOnMatch, options: QRegexOptions = {}) {
        pattern = new RegExp(pattern.source, pattern.flags);
        const spToken = options.special_token ?? "restart_regex";
        const global = pattern.flags.includes("g");
        let end = false;
        return new Q(data => {
            const { textSpecial, tokensRecieved, tokensRecievedNow } = data;
            let text = data.text ?? "";
            if (end) {
                return false;
            }
            let tokens = tokensRecieved;
            let packed = { content: data.content, text };
            if (tokensRecievedNow.some(e => e.special)) {
                const endIndex = tokensRecieved.length - tokensRecievedNow.length;
                const lastSpecialIndex = tokensRecieved.findLastIndex(e => e.special);
                const startIndex = lastSpecialIndex === -1 ? 0 : lastSpecialIndex + 1;
                const firstSpecialIndex = tokensRecievedNow.findIndex(e => e.special);
                const endIndexNew = firstSpecialIndex === -1 ? tokensRecievedNow.length : firstSpecialIndex;
                tokens = [...tokensRecieved.slice(startIndex, endIndex), ...tokensRecievedNow.slice(0, endIndexNew)];
                const packed = packTokens(tokens);
                text = packed.text ?? "";
            }
            try {
                const last = pattern.lastIndex;
                const m = pattern.exec(spToken === "include" ? textSpecial : text);
                if (m === null) {
                    pattern.lastIndex = last;
                    return false;
                } else {
                    if (!global) {
                        end = true;
                    }
                    const res = onMatch(data, m, Object.assign({ tokens }, packed)) ?? false;
                    if (typeof res === "boolean") {
                        return res;
                    } else if (res instanceof Q) {
                        return res.stopCondition(data);
                    } else {
                        throw new Error(`unexpected situation: expected bool|Q`);
                    }
                }
            } finally {
                if (tokensRecievedNow.some(e => e.special)) {
                    if (spToken === "restart_regex") {
                        pattern.lastIndex = 0;
                    } else if (spToken === "end_regex") {
                        end = true;
                    }
                }
            }
        });
    }
    public static substring(s: string, onMatch: QOnMatch, options: QSubstringOptions) {
        return this.regex(new RegExp(s.replaceAll(/[\\^$.|?*+()\[\]{}]/g, s => "\\" + s), options.ignore_case ? "gi" : "g"), onMatch, { special_token: options.special_token });
    }
    public static cond(stopCondition: Q["stopCondition"]) {
        return new Q(stopCondition);
    }
}







//