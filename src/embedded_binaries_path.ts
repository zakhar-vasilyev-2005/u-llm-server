import path from "path";


export function getPathToEmbeddedBinaries() {
    return path.join(path.dirname(import.meta.dirname), "binaries");
}
export function getPathToLlama(binariesDirectory: string) {
    return path.join(binariesDirectory, "llama-b9844", "libllama.so");
}
export function getPathToGGML(binariesDirectory: string) {
    return path.join(binariesDirectory, "llama-b9844", "libggml.so");
}
export function getPathToEntropy(binariesDirectory: string) {
    return path.join(binariesDirectory, "utils", "libentropy.so");
}
export function getPathToSamplingHelper(binariesDirectory: string) {
    return path.join(binariesDirectory, "utils", "libsamplinghelper.so");
}


