


export function repr(value: unknown) {
    return ["string", "number", "boolean"].some(e => typeof value === e) ? JSON.stringify(value) : String(value);
}