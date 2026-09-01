import { type SamplerConstructor } from "./llama-base.js";
import { SEventArgs } from './server-schemas.js';
import * as z from "zod";
import type { InferenceLineParams, InputElem, StopReason } from "./model.js";
import type { ContentElem, ModelClient, Token } from "./client.js";
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
export type TokensEvent = z.output<typeof SEventArgs["tokens"]>;
export type TokenSequence = {
    tokens: Token[],
    entropy: number,
    next: Token | null,
    stopReasons: StopReason[]
};
export type StopPredicateArg = {
    line: ClientLine,
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
    public async pull(stopCondition: StopCondition) {
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
                    line: this,
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
    public async resetConfig(...content: ContentElem[]) {
        const pieces = this.client.parse(...content);
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
        const elems = this.client.parse(...content);
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

export type CachedLinePullParams<T> = {
    query?: RQ<T> | undefined,
    sampler?: SamplerConstructor | undefined,
    inference: InferenceLineParams | undefined,
};
export class CachedLine {
    public tokens: Token[] = [];
    public allTokens: Token[] = [];
    public static async create(client: ModelClient, lineId?: string | undefined) {
        return new CachedLine(await ClientLine.create(client, lineId));
    }
    public constructor(public readonly origin: ClientLine) { }
    public async load(file: string) {
        await this.origin.loadContent(file);
        this.tokens = this.allTokens = this.origin.tokens;
    }
    public async save(file: string) {
        await this.origin.saveContent(file);
    }
    public async pull<T>(params: CachedLinePullParams<T>) {
        await this.cancel();
        await this.origin.setSampler(params.sampler ?? [{ type: "greedy" }], this.origin.tokens.length);
        const result = await (params.query ?? RQ.never()).pull(this.origin, params.inference);
        await this.origin.setSampler([{ type: "greedy" }], this.origin.tokens.length);
        return result;
    }
    public async step(...content: ContentElem[]) {
        const input = this.origin.client.parse(...content);
        let index = 0;
        let prefix = [] as { token?: number | undefined, piece: string, special: boolean }[];
        const shift = () => {
            const prefixElem = prefix.shift();
            if (prefixElem === undefined) {
                return this.allTokens[this.tokens.length + index++];
            } else {
                return prefixElem;
            }
        };
        const unshift = (e: { token?: number | undefined, piece: string, special: boolean }) => {
            const lastToken = this.allTokens[this.tokens.length + index - 1]?.token;
            if (lastToken !== undefined && e.token === lastToken) {
                index--;
            } else {
                prefix.unshift(e);
            }
        };
        while (true) {
            const token = shift();
            const elem = input.shift();
            if (elem === undefined) {
                if (token !== undefined) {
                    unshift(token);
                }
                break;
            } else {
                if (token === undefined) {
                    input.unshift(elem);
                    break;
                } else if (token.special && !elem.special) {
                    unshift(token);
                    input.unshift(elem);
                    break;
                } else if (elem.text.startsWith(token.piece)) {
                    elem.text = elem.text.slice(token.piece.length);
                    input.unshift(elem);
                    continue;
                } else if (token.piece.startsWith(elem.text)) {
                    unshift({ special: token.special || elem.special, piece: token.piece.slice(elem.text.length) });
                    continue;
                } else {
                    unshift(token);
                    input.unshift(elem);
                    break;
                }
            }
        }
        const keep = Math.max(0, index - prefix.length);
        this.tokens.push(...this.allTokens.slice(this.tokens.length, this.tokens.length + keep));
        await this.cancel();
        if (input.length !== 0) {
            await this.origin.step(...input);
            this.tokens = [...this.origin.tokens];
            this.allTokens = [...this.origin.tokens];
        }
    }
    public async cancel() {
        await this.origin.goto(this.tokens.length);
        this.allTokens = [...this.tokens];
    }
    public async goto(nTokens: number) {
        if (nTokens <= this.allTokens.length) {
            this.tokens = this.allTokens.slice(0, nTokens);
        } else {
            throw new Error(`cannot goto beyond currently generated amount of tokens`);
        }
    }
    public async clear() {
        await this.goto(0);
    }
    public async free() {
        await this.origin.free();
    }
}

export type RQRegexOptions = {
    special_token?: "restart_regex" | "end_regex" | "include" | undefined,
};
export type RQSubstringOptions = RQRegexOptions & {
    ignore_case?: boolean | undefined,
};
export type RQOnMatch<T> = (data: StopPredicateArg, m: RegExpExecArray, source: { tokens: Token[] } & PackedTokens) => RQ<T> | T | undefined;
export type RQOnEvery<T, P extends RQ<unknown>[] | []> = (data: StopPredicateArg, results: RQResultsOfEvery<P>) => RQ<T> | T | undefined;
export type RQCallback<T> = (data: StopPredicateArg) => RQ<T> | T | undefined;
export type RQResult<T> = {
    reason: T,
    result: PullResult,
};
export type RQResultInference = {
    reason?: undefined,
    result: PullResult,
};
export type RQType<T extends RQ<unknown>> = T extends RQ<infer V> ? V : never;
export type RQTypeList<T extends RQ<unknown>[] | []> = { [k in keyof T]: RQType<T[k]> } & unknown[];
export type RQResultsOfEvery<P extends RQ<unknown>[] | []> = { [k in keyof P]: P[k] extends RQ<infer V> ? Omit<RQResult<V>, "result"> : never } & unknown[];
export class RQ<const T> {
    public static stoppedByInferenceParams = Symbol("reasonNotSet");
    public constructor(public readonly cb: RQCallback<T>) { }
    public static never() {
        return this.cond(() => undefined) as RQ<never>;
    }
    public static some<const T extends RQ<unknown>[] | []>(conditions: T): RQ<RQTypeList<T>[number]> {
        return new RQ(function (data) {
            for (const rq of conditions as RQ<RQTypeList<T>[number]>[]) {
                const res = rq.run(data);
                if (res !== null) {
                    return res;
                }
            }
            return;
        });
    }
    public static every<const T, const P extends RQ<unknown>[] | []>(conditions: P, cb: RQOnEvery<T, P>): RQ<T> {
        return new RQ(function (data) {
            const results = conditions.map(e => e.run(data));
            return results.every(e => e !== null) ? cb(data, results as any) : undefined;
        });
    }
    public static regex<const T>(pattern: RegExp, onMatch: RQOnMatch<T>, options: RQRegexOptions = {}): RQ<T> {
        pattern = new RegExp(pattern.source, pattern.flags);
        const spToken = options.special_token ?? "restart_regex";
        const global = pattern.flags.includes("g");
        let lastMatch: null | RegExpExecArray = null;
        let end = false;
        return this.cond(function (data) {
            const { textSpecial, tokensRecieved, tokensRecievedNow } = data;
            let text = data.text ?? "";
            if (end) {
                return;
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
                lastMatch = pattern.exec(spToken === "include" ? textSpecial : text);
                if (lastMatch === null) {
                    pattern.lastIndex = last;
                    return;
                } else {
                    if (!global) {
                        end = true;
                    }
                    return onMatch(data, lastMatch, Object.assign({ tokens }, packed));
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
    public static substring<const T>(s: string, onMatch: RQOnMatch<T>, options: RQSubstringOptions = {}): RQ<T> {
        return this.regex(new RegExp(s.replaceAll(/[\\^$.|?*+()\[\]{}]/g, s => "\\" + s), options.ignore_case ? "gi" : "g"), onMatch, { special_token: options.special_token });
    }
    public static eog<const T>(cb: RQCallback<T>): RQ<T> {
        return this.cond(function (data) {
            return (data.next !== null && !!data.line.client.modelTokenInfo[data.next.token]?.eog) ? cb(data) : undefined;
        });
    }
    public static tokens<const T>(cond: (tokens: number) => boolean, cb: RQCallback<T>): RQ<T> {
        return this.cond(function (data) {
            return cond(data.tokensRecieved.length) ? cb(data) : undefined;
        });
    }
    public static symbols<const T>(cond: (symbols: number) => boolean, cb: RQCallback<T>): RQ<T> {
        return this.cond(function (data) {
            return cond(data.text?.length ?? 0) ? cb(data) : undefined;
        });
    }
    public static symbolsAll<const T>(cond: (symbols: number) => boolean, cb: RQCallback<T>): RQ<T> {
        return this.cond(function (data) {
            return cond(data.textSpecial.length) ? cb(data) : undefined;
        });
    }
    public static cond<const T extends RQCallback<K>, const K>(cb: T): RQ<T extends RQCallback<infer V> ? V : never> {
        return new RQ(cb) as any;
    }
    public async pull(line: ClientLine, inferenceParams: InferenceLineParams = {}): Promise<RQResult<T> | RQResultInference> {
        let reason: T | undefined = undefined;
        const result = await line.pull(Object.assign({
            stop_predicate: (data: StopPredicateArg) => {
                reason = this.run(data);
                return reason !== undefined;
            }
        }, inferenceParams));
        return { reason, result };
    }
    public run(data: StopPredicateArg): T | undefined {
        const res = this.cb(data);
        return res instanceof RQ ? this.run(data) : res;
    }
}





//