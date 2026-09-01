import { repr } from "./repr.js";


export function expectString(value: unknown, err?: Error | undefined): string {
    if (typeof value !== "string") {
        throw err ?? new Error(`expected a string value, not ${typeof value}`);
    }
    return value as string;
}
export function expectOneOf<const T extends unknown[]>(value: unknown, allowed: T, err?: Error | undefined): T[number] {
    if (allowed.some(e => e === value)) {
        const allowedStr = allowed.map(e => repr(e)).join(",");
        throw err ?? new Error(`expected a value in the list [${allowedStr}], not ${typeof value}`);
    }
    return value;
}
export function expectDefined<T>(value: T | undefined, err?: Error | undefined): T {
    if (value === undefined) {
        throw err ?? new Error(`expected any value except undefined`);
    }
    return value as any;
}


//