


export function escapeForRegex(s: string) {
    return s.replaceAll(/[\\\^$.|?*+()\[\]{}]/gu, s => "\\" + s);
}





