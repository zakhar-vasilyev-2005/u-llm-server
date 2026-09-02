import { Yurandom } from "yurandom/index.js";
import { blendObjects } from "./typeutils.js";
import type { SamplerConstructor } from "./llama-base.js";





export function samplerWithSeed(sampler: SamplerConstructor, seedOrRNG: number | string | Yurandom): SamplerConstructor {
    const intSeed = typeof seedOrRNG === "number" ? seedOrRNG : (typeof seedOrRNG === "string" ? new Yurandom(seedOrRNG) : seedOrRNG).int(1, 32000);
    return sampler.map(e => {
        return (e as { seed?: unknown }).seed === undefined ? e : blendObjects(e, { seed: intSeed });
    });
}





//