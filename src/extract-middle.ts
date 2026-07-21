


export function extractMiddle(text: string, start: string, end: string) {
    const firstSplit = text.split(start);
    if (firstSplit.length > 2) { return undefined; }
    const secondSplit = (firstSplit.at(-1) as string).split(end);
    if (secondSplit.length > 2) { return undefined; }
    return secondSplit[0] as string;
};
