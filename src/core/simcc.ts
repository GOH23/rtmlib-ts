/**
 * SimCC (Simulated Coordinate Classification) decoder helpers.
 *
 * RTMW / RTMW3D emit a per-keypoint soft-argmax vector over a 1D grid;
 * `argmaxSimCC` returns both the integer argmax position and the
 * confidence at that bin. The 3D pose path uses a slightly different
 * axis layout and stays inline in `pose3dDetector.ts`.
 */
export interface SimCCResult {
  argmax: number;
  max: number;
}

/**
 * Argmax over a slice of `s` starting at `base`, of length `len`.
 * Returns the bin index (relative to the full array, not the slice)
 * and the value at that bin.
 */
export function argmaxSimCC(s: Float32Array, base: number, len: number): SimCCResult {
  let argmax = base;
  let max = s[base];
  for (let i = base + 1, end = base + len; i < end; i++) {
    const v = s[i];
    if (v > max) {
      max = v;
      argmax = i;
    }
  }
  return { argmax, max };
}
