import type { LibLlama } from "./llama-base.js";


export type TokenInfo = {
    token: number,
    pieceBytesBase64: string,
    pieceText: string,
    eog: boolean,
    special: boolean,
};

export function tokeninfoBase(llama: LibLlama, vocabPtr: bigint, startIndex: number, endIndex: number) {
    const result: TokenInfo[] = [];
    for (let i = startIndex; i < endIndex; i++) {
        const piece = llama.token_to_piece_raw(vocabPtr, i, true);
        result.push({
            token: i,
            pieceBytesBase64: piece.toString("base64"),
            pieceText: piece.toString("utf8"),
            eog: llama.vocab_is_eog(vocabPtr, i),
            special: llama.vocab_is_control(vocabPtr, i),
        });
    }
    return Object.fromEntries(result.map(e => [e.token, e]));
}




