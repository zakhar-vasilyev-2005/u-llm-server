import { expectDefined } from "./expect.js";


export type Defined<T> = T extends (infer P) | undefined ? P : T;
export type PromiseOrNot<T> = Promise<T> | T;
export type AllKeys<T> = T extends any ? keyof T : never;
export type UnknownRecord = Record<string | symbol | number, unknown>;
export type AnyRecord = Record<string | symbol | number, any>;
export type BlendObjects<T extends object[]> = (
    T extends [infer Head, ...(infer Tail extends object[])]
    ? Omit<Head, keyof BlendObjects<Tail>> & BlendObjects<Tail>
    : {}
);
export type BlendObjectsAK<T extends object[]> = (
    T extends [infer Head, ...(infer Tail extends object[])]
    ? Omit<Head, AllKeys<BlendObjects<Tail>>> & BlendObjects<Tail>
    : {}
);
export type OmitEvery<T extends object, K extends (string | number | symbol)[]> = (
    K extends [infer Head extends (string | number | symbol), ...(infer Tail extends (string | number | symbol)[])]
    ? OmitEvery<Omit<T, Head>, Tail>
    : T
);
export type DeepReplaceField<T extends object, K extends string | number | symbol, V> = {
    [k in keyof T]: k extends K ? V : (T[k] extends object ? DeepReplaceField<T[k], K, V> : T[k]);
}
export type DeepReplaceFieldAK<T extends object, K extends string | number | symbol, V> = {
    [k in AllKeys<T>]: k extends K ? V : (T[k] extends object ? DeepReplaceField<T[k], K, V> : T[k]);
}


export type OrType<T extends object, P> = {
    [k in keyof T]: T[k] | P
};
export type OrUndefined<T extends object> = {
    [k in keyof T]?: T[k] | undefined
};
export type StripType<T extends object, P> = {
    [k in keyof T]: T[k] extends (infer V | P) ? V : never
};
export type StripUndefined<T extends object> = StripType<T, undefined>;
export type ReplaceType<T extends object, V> = {
    [k in keyof T]: V
};


export type AKOrType<T extends object, P> = {
    [k in AllKeys<T>]: T[k] | P
};
export type AKOrUndefined<T extends object> = {
    [k in AllKeys<T>]?: T[k] | undefined
};
export type AKStripType<T extends object, P> = {
    [k in AllKeys<T>]: T[k] extends (infer V | P) ? V : never
};
export type AKStripUndefined<T extends object> = AKStripType<T, undefined>;
export type AKReplaceType<T extends object, V> = {
    [k in AllKeys<T>]: V
};


export function isObject<const T>(object: T): T extends object ? true : false {
    return ((typeof object === "object" || typeof object === "function") && object !== null) as any;
}
export function asItself<const T>(object: T): T {
    return object;
}
export function stripUndefined<T extends object>(object: T): StripUndefined<T> {
    const result: UnknownRecord = {};
    for (const k of [...Object.getOwnPropertyNames(object), ...Object.getOwnPropertySymbols(object)]) {
        if ((object as UnknownRecord)[k] !== undefined) {
            result[k] = (object as UnknownRecord)[k];
        }
    }
    return result as any;
}
export function stripFields<T extends object, const K extends (string | number | symbol)[]>(object: T, ...keys: K): OmitEvery<T, K> {
    const result = Object.assign({}, object) as UnknownRecord;
    for (const key of keys) {
        if (key in result) {
            delete result[key];
        }
    }
    return result as any;
}

export function blendObjects<const T extends object[]>(...objects: T): BlendObjects<T> {
    return assignAll({}, ...objects) as any;
}
export function blendObjectsAK<const T extends object[]>(...objects: T): BlendObjectsAK<T> {
    return assignAll({}, ...objects) as any;
}
export function assignAll<const B extends object, const T extends object[]>(base: B, ...objects: T): BlendObjects<[B, ...T]> {
    let temp: object = base;
    for (const object of objects) {
        temp = Object.assign(temp, object);
    }
    return temp as any;
}
export function assignAllAK<const B extends object, const T extends object[]>(base: B, ...objects: T): BlendObjectsAK<[B, ...T]> {
    return assignAll(base, ...objects) as any;
}

export function objectMap<const T extends object, const R>(object: T, cb: <K extends keyof T>(value: T[K], key: K, object: T) => R): { [k in keyof T]: R } {
    return Object.fromEntries(Object.entries(object).map(([k, v]) => [k, cb(v, k as any, object)] as [string, any])) as any;
}
export async function objectMapAsync<const T extends object, const R>(object: T, cb: <K extends keyof T>(value: T[K], key: K, object: T) => PromiseOrNot<R>): Promise<{ [k in keyof T]: R }> {
    return Object.fromEntries(await Promise.all(Object.entries(object).map(async ([k, v]) => [k, await cb(v, k as any, object)] as [string, any]))) as any;
}

export function objectMapAK<const T extends object, const R>(object: T, cb: <K extends AllKeys<T>>(value: T[K], key: K, object: T) => R): { [k in AllKeys<T>]: R } {
    return objectMap(object, cb as any) as any;
}
export async function objectMapAsyncAK<const T extends object, const R>(object: T, cb: <K extends AllKeys<T>>(value: T[K], key: K, object: T) => PromiseOrNot<R>): Promise<{ [k in AllKeys<T>]: R }> {
    return objectMapAsync(object, cb as any) as any;
}

export async function objectPromiseAllAK<const T extends object>(object: T): Promise<{ [k in AllKeys<T>]: Awaited<T[k]> }> {
    return objectPromiseAll(object);
}
export async function objectPromiseAll<const T extends object>(object: T): Promise<{ [k in keyof T]: Awaited<T[k]> }> {
    return Object.fromEntries(await Promise.all(Object.entries(object).map(([k, v]) => (async () => v)().then(r => [k, r] as [string, unknown])))) as any;
}






//