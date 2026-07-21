import fs from 'fs-extra';
import { parseINI, parseJSON5, parseTOML, parseYAML } from 'confbox'


export function readConfig(file: string) {
    const content = fs.readFileSync(file, { encoding: "utf8" });
    if (file.startsWith(".")) { file = file.slice(1); }
    const parts = file.split(".");
    const ext = parts.length < 2 ? ".ini" : "." + parts.at(-1);
    const parse = {
        ".ini": parseINI,
        ".cfg": parseINI,
        ".conf": parseINI,
        ".properties": parseINI,
        ".json": parseJSON5,
        ".jsonc": parseJSON5,
        ".json5": parseJSON5,
        ".yaml": parseYAML,
        ".yml": parseYAML,
        ".toml": parseTOML,
    }[ext] as (s: string) => any;
    return parse(content);
}