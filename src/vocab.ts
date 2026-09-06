import type { Token } from "./client.js";
import { LibLlama } from "./llama-base.js";




export class ModelVocab {
    public readonly file: string;
    public readonly ptr: bigint;
    public readonly size: number;
    public readonly llama: LibLlama;
    public readonly special: {
        bos: Token | null,
        eos: Token | null,
        eot: Token | null,
        fim_mid: Token | null,
        fim_pad: Token | null,
        fim_pre: Token | null,
        fim_rep: Token | null,
        fim_sep: Token | null,
        fim_suf: Token | null,
        add_bos: boolean,
        add_eos: boolean,
        add_sep: boolean,
    };
    public constructor(llamaLibPath: string, vocabFile: string) {
        this.llama = new LibLlama(llamaLibPath);
        this.file = vocabFile;
        const mparams = this.llama.model_default_params();
        mparams.vocab_only = true;
        mparams.use_mmap = true;
        const modelPtr = this.llama.model_load_from_file(vocabFile, mparams);
        if (modelPtr === null) {
            throw new Error(`cannot load vocab from file ${JSON.stringify(vocabFile)}`);
        }
        this.ptr = this.llama.model_get_vocab(modelPtr);
        this.size = this.llama.vocab_n_tokens(this.ptr);
        this.special = {
            bos: this.parse(this.llama.vocab_bos(this.ptr), true),
            eos: this.parse(this.llama.vocab_eos(this.ptr), true),
            eot: this.parse(this.llama.vocab_eot(this.ptr), true),
            fim_mid: this.parse(this.llama.vocab_fim_mid(this.ptr), true),
            fim_pad: this.parse(this.llama.vocab_fim_pad(this.ptr), true),
            fim_pre: this.parse(this.llama.vocab_fim_pre(this.ptr), true),
            fim_rep: this.parse(this.llama.vocab_fim_rep(this.ptr), true),
            fim_sep: this.parse(this.llama.vocab_fim_sep(this.ptr), true),
            fim_suf: this.parse(this.llama.vocab_fim_suf(this.ptr), true),
            add_bos: this.llama.vocab_get_add_bos(this.ptr),
            add_eos: this.llama.vocab_get_add_eos(this.ptr),
            add_sep: this.llama.vocab_get_add_sep(this.ptr),
        };
    }
    public parse(token: number, ignoreError?: false): Token;
    public parse(token: number, ignoreError: true): Token | null;
    public parse(token: number, ignoreError: boolean = false) {
        if (!this.isValid(token) && ignoreError) {
            return null;
        }
        return { token, piece: this.toPiece(token).toString(), special: this.isSpecial(token) } as Token;
    }
    public isValid(token: number) {
        return Number.isInteger(token) && token >= 0 && token < this.size;
    }
    public validateToken(token: number) {
        if (!Number.isInteger(token) || token < 0 || token >= this.size) {
            throw new Error(`token ${token} does not fit in vocab range (integer from 0 to ${this.size - 1} inclusive)`);
        }
        return token;
    }
    public validate<T extends number[] | Int32Array>(tokens: T): T {
        const index = tokens instanceof Int32Array ? tokens.findIndex(e => e < 0 || e >= this.size) : tokens.findIndex(e => !Number.isInteger(e) || e < 0 || e >= this.size);
        if (index !== -1) {
            throw new Error(`token #${index} (${tokens[index]}) does not fit in vocab range (integer from 0 to ${this.size - 1} inclusive)`);
        }
        return tokens;
    }
    public isSpecial(token: number) {
        return this.llama.vocab_is_control(this.ptr, this.validateToken(token));
    }
    public isEOG(token: number) {
        return this.llama.vocab_is_eog(this.ptr, this.validateToken(token));
    }
    public toPiece(token: number, renderSpecial: boolean = true) {
        return this.llama.token_to_piece_raw(this.ptr, this.validateToken(token), renderSpecial);
    }
    public tokenize(text: string, parseSpecial: boolean, addSpecial: boolean = false) {
        return this.llama.tokenize(this.ptr, text, parseSpecial, addSpecial);
    }
    public detokenize(tokens: number[] | Int32Array, unparseSpecial: boolean, removeSpecial: boolean = false) {
        return this.llama.detokenize(this.ptr, this.validate(tokens), unparseSpecial, removeSpecial);
    }
}













//