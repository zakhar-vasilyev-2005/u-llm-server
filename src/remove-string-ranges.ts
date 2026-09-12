import { escapeForRegex } from "./escape-for-regex.js";

export function removeRanges(text: string, startMarker: string, endMarker: string) {
    const patternAll = new RegExp(`(${escapeForRegex(startMarker)})([\s\S]*?)(${escapeForRegex(endMarker)})`, "gu");
    const patternBoundary = new RegExp(`(^${escapeForRegex(endMarker)})|(${escapeForRegex(startMarker)}$)`, "gu");
    text = startMarker + text + endMarker;
    for (const pattern of [patternAll, patternBoundary]) {
        while (true) {
            const newText = text.replaceAll(pattern, "");
            if (text === newText) {
                break;
            } else {
                text = newText;
                continue;
            }
        }
    }
    return text;
}

