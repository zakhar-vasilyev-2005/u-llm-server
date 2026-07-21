


export class AtomicFlag {
    public readonly shared: SharedArrayBuffer;
    public readonly array: Uint32Array;
    public constructor(shared: SharedArrayBuffer | null = null) {
        this.shared = shared ?? new SharedArrayBuffer(4);
        this.array = new Uint32Array(this.shared);
    }
    public set(value: boolean | number) {
        if (typeof value === "boolean") { value = value ? 1 : 0; }
        Atomics.store(this.array, 0, value);
    }
    public get() {
        return Atomics.load(this.array, 0);
    }
}





