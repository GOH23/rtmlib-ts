/**
 * Tiny stats helper. `attachStats(arr, stats)` stashes an arbitrary
 * stats object on a result array without the `(arr as any).stats = …`
 * cast each detector used to write.
 */
export function attachStats<T>(arr: T[], stats: unknown): T[] {
  (arr as unknown as { stats: unknown }).stats = stats;
  return arr;
}
