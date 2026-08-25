import { LibLlama } from "./llama-base.js";
import { tokeninfoBase } from "./tokeninfo-base.js";
import { expose, getParent } from "./worker.js";



export type Args = {
    llama_library: string
};
export type Events = {};
export type API = {
    tokeninfo: typeof tokeninfo,
};


const llama = new LibLlama(getParent<Events, Args>().args.llama_library);
llama.log_set(() => { });

export function tokeninfo(vocabPtr: bigint, start: number = 0, end: number = Number.POSITIVE_INFINITY) {
    const size = llama.vocab_n_tokens(vocabPtr);
    start = start <= 0 ? 0 : start >= size ? size : start;
    end = end <= 0 ? 0 : end >= size ? size : end;
    return tokeninfoBase(llama, vocabPtr, start, end);
}
expose<API>({ tokeninfo });




