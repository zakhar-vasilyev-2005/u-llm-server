


export type GrowBufferStrategy = (current: number, required: number) => number;
export class GrowBuffer {
    public static defaultStrategy: GrowBufferStrategy = (a, b) => b * 1.5 + 200;
    public capacity: number = 0;
    public fullBuffer: Buffer = Buffer.alloc(0)
    public buffer: Buffer = this.fullBuffer.subarray(0, 0);
    public length: number = 0;
    public constructor(baseCapacity: number = 1024, public strategy: GrowBufferStrategy = GrowBuffer.defaultStrategy) {
        this.clear(baseCapacity);
    }
    public static from(buffer: Buffer) {
        const growbuffer = new GrowBuffer(GrowBuffer.defaultStrategy(buffer.byteLength, buffer.byteLength));
        growbuffer.push(buffer);
        return growbuffer;
    }
    public push(buffer: Buffer, start: number = 0, end: number = buffer.length) {
        this.expandTo(this.length + buffer.length);
        this.updateBuffer(this.length + buffer.length);
        buffer.copy(this.buffer, this.length - buffer.length, start, end);
    }
    public calcStrategy(newSize: number, currentCapacity: number = this.capacity) {
        return Math.max(newSize, Math.floor(this.strategy(currentCapacity, newSize)))
    }
    public expandTo(newSize: number) {
        if (this.capacity >= newSize) { return; }
        const newCapacity = this.calcStrategy(newSize);
        const newBuffer = Buffer.alloc(newCapacity);
        this.buffer.copy(newBuffer);
        this.fullBuffer = newBuffer;
        this.capacity = newCapacity;
        this.updateBuffer();
    }
    public expand(size: number) { this.expandTo(this.length + size); }
    public shrinkTo(newSize: number) { this.updateBuffer(newSize); }
    public shrink(size: number) { this.updateBuffer(this.length - size); }
    public replaceWith(buffer: Buffer, start: number = 0, end = buffer.length) {
        this.shrinkTo(0);
        this.push(buffer, start, end);
    }
    private updateBuffer(newLength: number = this.length) {
        this.length = Math.max(0, newLength);
        this.buffer = this.fullBuffer.subarray(0, this.length);
    }
    public copy() {
        const copy = Buffer.alloc(this.buffer.byteLength);
        this.buffer.copy(copy);
        return copy;
    }
    public clear(newCapacity: number = 1024) {
        this.fullBuffer = Buffer.alloc(newCapacity);
        this.capacity = newCapacity;
        this.updateBuffer(0);
    }
}



