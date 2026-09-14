export const LIBRARY_PAGE_SIZE = 20

export function libraryPageCount(total: number, pageSize = LIBRARY_PAGE_SIZE): number {
    return Math.max(1, Math.ceil(Math.max(0, total) / pageSize))
}

export function libraryRange(
    page: number,
    count: number,
    total: number,
    pageSize = LIBRARY_PAGE_SIZE,
): string {
    if (!total || !count) return '0 games'
    const start = page * pageSize + 1
    return `${start}–${start + count - 1} of ${total}`
}
