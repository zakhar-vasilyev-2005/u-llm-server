import { EventEmitter } from "events";
import { Worker as ClassicWorker, parentPort, workerData, type WorkerOptions } from "worker_threads";

export type APIBase = Record<string, (...args: any[]) => any | Promise<any>>;
export type EventsBase = Record<string, any[]>;
export type ArgsBase = Record<string, any>;
export type WorkerAPI<API extends APIBase> = {
    [k in keyof API]: (...args: Parameters<API[k]>) => Promise<Awaited<ReturnType<API[k]>>>
};
export class Worker<API extends APIBase, Events extends EventsBase> extends EventEmitter<Events> {
    public constructor(public readonly worker: ClassicWorker, public readonly api: WorkerAPI<API>) { super(); }
    public async waitEvent<EventName extends keyof Events>(name: EventName): Promise<Events[EventName]>;
    public async waitEvent(filter: <Name extends keyof Events>(eventName: Name, ...eventArgs: Events[Name]) => boolean): Promise<Events[keyof Events]>;
    public async waitEvent<EventName extends keyof Events>(
        filter: EventName | (<Name extends keyof Events>(eventName: Name, ...eventArgs: Events[Name]) => boolean)
    ): Promise<Events[keyof Events]> {
        if (typeof filter !== "function") {
            const eventName = filter as keyof Events;
            filter = (name: keyof Events, ...args: any[]) => name === eventName;
        }
        return await new Promise<Events[keyof Events]>(resolve => {
            const handler = (msg: any) => {
                if (typeof msg !== "object" || msg === null || !("event" in msg)) { return; }
                if (!filter(msg["event"], ...msg["args"] ?? [])) { return; }
                this.worker.off("message", handler);
                resolve(msg["args"] as Events[keyof Events]);
            };
            this.worker.on("message", handler);
        });
    }
    public static async start<API extends APIBase, Events extends EventsBase, Args extends ArgsBase>(filename: string, args: Args, options: WorkerOptions = {}) {
        const worker = new ClassicWorker(filename, Object.assign(options, { workerData: args }));
        const funcNames = await new Promise<string[]>(resolve => {
            const handler = (msg: any) => {
                if (typeof msg !== "object" || msg === null || !("api" in msg)) { return; }
                worker.off("message", handler);
                let api = msg["api"]
                if (!(api instanceof Array)) { api = [api]; }
                resolve([...api].filter(e => typeof e === "string"));
            };
            worker.on("message", handler);
            worker.postMessage("api");
        });
        let counter = 0;
        const api = Object.fromEntries(funcNames.map(name => [
            name as keyof API,
            (...args: any[]) => {
                const id = String(counter++);
                // if this returns an error, it's an error from worker
                return new Promise((resolve, reject) => {
                    const handler = (msg: any) => {
                        if (typeof msg !== "object" || msg === null || msg.id !== id) { return; }
                        worker.off("message", handler);
                        if ("error" in msg) {
                            const params = msg["error"];
                            const message = params.message;
                            delete params.message;
                            reject(Object.assign(new Error(message), params));
                        } else {
                            resolve(msg["result"]);
                        }
                    };
                    worker.on("message", handler);
                    worker.postMessage({ exec: name, id, args });
                });
            }
        ])) as WorkerAPI<API>;
        const result = new Worker<API, Events>(worker, api);
        worker.on("message", msg => {
            if (typeof msg !== "object" || msg === null || !("event" in msg)) { return; }
            const event = msg["event"];
            let args = msg["args"] ?? [];
            if (!(args instanceof Array)) { args = [args]; }
            result.emit(event, ...args);
        });
        return result;
    }
}

export const parentPortData = new WeakMap<MessagePort, {
    to_exit: boolean
}>();
export function expose<API extends APIBase>(api: API) {
    if (parentPort === null) { return; }
    const portEntry = { to_exit: false };
    parentPortData.set(parentPort, portEntry);
    parentPort.on("message", msg => {
        if (msg === "api") {
            parentPort?.postMessage({ api: Object.keys(api) });
        }
    })
    parentPort.on("message", async msg => {
        if (typeof msg !== "object" || msg === null || !("exec" in msg)) { return; }
        const { exec, id, args } = msg;
        try {
            const func = api[exec];
            if (func === undefined) { return; }
            const realArgs = args instanceof Array ? [...args] : [args];
            const result = await func(...realArgs);
            parentPort?.postMessage({ result, id });
            if (portEntry.to_exit) {
                parentPort?.removeAllListeners();
                parentPort?.close();
            }
        } catch (e) {
            const error = Object.assign(Object.assign({ message: "" }, e), { message: e instanceof Error ? e.message : `Unknown error: ${e}` });
            parentPort?.postMessage({ error, id });
        }
    });
}

export function getParent<Events extends EventsBase, Args extends ArgsBase>() {
    const emit = function <Key extends keyof Events>(name: Key, ...args: Events[Key]) {
        if (parentPort === null) { throw new Error(`can only 'emit' if in worker thread`); }
        parentPort.postMessage({ event: name, args });
    };
    const exit = () => {
        const entry = parentPort && parentPortData.get(parentPort);
        if (!!entry) {
            entry.to_exit = true;
        }
    };
    return { emit, args: workerData as Args, exit };
}




//