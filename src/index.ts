import path from "path";
import { type ModelParamsSerialized } from "./llama-base.js";
import { ModelServer, type ConnOption } from "./service.js";
import { Argument, Command, Option } from 'commander';
import { Yurandom } from "yurandom/index.js";

export * from './llama.js';
export * from './model.js';
export * from './llama-base.js';
export * from './service.js';
export * from './server-schemas.js';


export const DEFAULT_PORT = 37213;
export const program = new Command();
program
    .name('u-llm-server')
    .description('A llama.cpp-based inference server with slightly extended capabilities in realtime inference control.')
    .version('1.0.0')
    .addOption(
        new Option('-p, --port [<host>:]<port>', ` a host and port to listen on; can be shortened to just a port; if no connections specified, then localhost:${DEFAULT_PORT} is used`)
            .argParser((a, b) => {
                const parts = a.split(":");
                const port = parts.pop() as string;
                const host = parts.join(":") || "localhost";
                if (/^\d+$/.exec(port) === null || parseInt(port) > 65535) {
                    process.stderr.write(`error: bad port given in '${a}': expected an integer between 0 and 65535\n`);
                    process.exit(5);
                }
                return [{ host, port: parseInt(port) }, ...(b instanceof Array ? b : b === undefined ? [] : [b])];
            })
    )
    .addOption(
        new Option('-u, --unix <socket>', ' a path to a unix socket to listen on; works in addition to --port')
            .argParser((file, b) => {
                return [{ unix: file }, ...(b instanceof Array ? b : b === undefined ? [] : [b])];
            })
    )
    .addOption(
        new Option('-l, --ngl <number>', ' a number of model\'s layers to offload into gpu')
            .argParser(raw => {
                if (/^\d+$/.exec(raw) === null) {
                    process.stderr.write(`error: --ngl param must have an integer, nonnegative value`);
                    process.exit(7);
                }
                return parseInt(raw);
            })
    )
    .addOption(
        new Option('-g, --main-gpu <number>', ' an id of main gpu to use')
            .argParser(raw => {
                if (/^\d+$/.exec(raw) === null) {
                    process.stderr.write(`error: --main-gpu param must have an integer, nonnegative value`);
                    process.exit(8);
                }
                return parseInt(raw);
            })
    )
    .addOption(
        new Option('-s, --split-mode <type>', " determines how to split model's tensors across the gpus and cpu; available modes are: 'none', 'layer' (default), 'row', 'tensor'")
            .argParser(raw => {
                if (["none", "layer", "row", "tensor"].find(e => e === raw) === undefined) {
                    process.stderr.write(`error: --split-mode param got bad value: '${raw}', see --help for available values`);
                    process.exit(9);
                }
                return raw;
            })
    )
    .addArgument(new Argument('<model>', ' a gguf-file to read model from'))
    .action(async (model, { port, unix, ngl, mainGpu, splitMode }) => {
        const conn = [...port ?? [], ...unix ?? []] as ConnOption[];
        if (conn.length === 0) {
            conn.push({ host: "localhost", port: DEFAULT_PORT });
        }
        const mparams = {
            n_gpu_layers: ngl ?? 999,
            main_gpu: mainGpu ?? 0,
            split_mode: splitMode ?? "layer",
        } as ModelParamsSerialized;
        try {
            await main(model, mparams, conn);
        } catch (e) {
            process.stderr.write(`error: ${String(e)}`);
            const seed = Object.prototype.toString.call(e);
            const rng = new Yurandom(seed);
            process.exit(rng.int(10, 50));
        }
    });


export async function main(modelFile: string, modelParams: ModelParamsSerialized, connections: ConnOption[]) {
    const server = new ModelServer(modelFile, modelParams);
    await Promise.all(connections.map(conn => server.listen(conn)));
    const exitCb = () => {
        server.close();
        process.exit();
    };
    process.on("SIGTERM", exitCb);
    process.on("SIGINT", exitCb);
    await Promise.all([
        new Promise(resolve => {
            server.on("close", () => resolve(undefined));
        }),
        new Promise(resolve => {
            server.run(path.join(path.dirname(import.meta.dirname), "binaries"));
            resolve(undefined);
        })
    ]);
}



