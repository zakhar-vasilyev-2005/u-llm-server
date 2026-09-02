

export type PromiseOrNot<T> = Promise<T> | T;
export type AllKeys<T> = T extends any ? keyof T : never;
export type UnknownRecord = Record<string | symbol | number, unknown>;
export type AnyRecord = Record<string | symbol | number, any>;
export type BlendObjects<T extends object[]> = (
    T extends [infer Head, ...(infer Tail extends object[])]
    ? Head & BlendObjects<Tail>
    : {}
);
export type OmitEvery<T extends object, K extends (string | number | symbol)[]> = (
    K extends [infer Head extends (string | number | symbol), ...(infer Tail extends (string | number | symbol)[])]
    ? OmitEvery<Omit<T, Head>, Tail>
    : T
);


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
    let temp: object = {};
    for (const object of objects) {
        temp = Object.assign(temp, object);
    }
    return temp as any;
}





//