import { main, ModelParamsSchema } from './index.js';

const [modelFile, connStringified, modelParamsStringified] = process.argv.slice(2);
if (modelFile === undefined) {
    console.error(`missing 'modelFile' argument`);
    process.exit(1);
}
let connRaw, modelParamsRaw;
try {
    connRaw = JSON.parse(connStringified ?? "[]");
    modelParamsRaw = JSON.parse(modelParamsStringified ?? "{}");
} catch (e) {
    console.error("JSON error while parsing argv:", e);
    process.exit(1);
}
const conn = (connRaw instanceof Array ? connRaw : [connRaw]).map(e =>
    "unix" in e ?
        { unix: e.unix } :
        "port" in e ?
            { port: parseInt(e.port), host: String(e.host ?? "localhost") } :
            null
).filter(e => e !== null);
const modelParamsParmsed = ModelParamsSchema.safeParse(modelParamsRaw);
if (!modelParamsParmsed.success) {
    console.error("Schema error while parsing 'modelParams' argument:", modelParamsParmsed.error.issues);
    process.exit(1);
}
process.stdout.on("error", () => { });
process.stderr.on("error", () => { });
try {
    await main(modelFile, modelParamsParmsed.data, conn);
} catch (e) {
    console.error("Unknown error while server running:", e);
    process.exit(1);
}


//