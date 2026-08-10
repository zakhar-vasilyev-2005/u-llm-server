import { exec } from 'child_process';
import { ClientLine, ModelClient } from './server.js';

const client = await ModelClient.create({
    conn: { port: 32256 },
    timeout: 1,
    fallbackStartServer: {
        modelFile: "/mnt/120gb/Users/Public/LLMs/Llama-3.2-1B-Instruct-IQ4_XS.gguf",
        modelParams: {
            check_tensors: false,
            main_gpu: 1,
            n_gpu_layers: 999,
            split_mode: "none",
        },
        stderr: "inherit",
        stdout: "inherit",
        timeout: 100_000,
    }
});
try {
    const pre = client.prefixes;
    const line = await ClientLine.create(client, "main", [{ type: "dist", seed: 1 }]);
    const p1 = await line.step(pre.initToUser, "Hello?", pre.userToAssistant);
    const out = await line.pull({ max_tokens: 10, eog_stop: true });
    console.log(out);
    await line.goto(p1);
    await line.setSampler([{ type: "dist", seed: 2 }]);
    const out2 = await line.pull({ max_tokens: 10, eog_stop: true });
    console.log(out2);
} finally {
    await client.close();
}




//