/**
 * Generic Non-Maximum Suppression.
 *
 * YOLO-style detectors emit hundreds of overlapping candidate boxes.
 * NMS picks the highest-scoring box per cluster by sorting by score
 * and discarding every later box whose IoU with an already-kept box
 * exceeds `iouThreshold`. Used by every multi-class detector after
 * the score + class-id filter.
 */

export interface NmsItem {
  bbox: { x1: number; y1: number; x2: number; y2: number };
  score: number;
}

/**
 * Suppress overlapping boxes. Items must expose `bbox` (xyxy) and
 * `score` (higher = more confident). Returns a new array containing
 * only the surviving items in descending-score order — the caller's
 * original array is not mutated.
 */
export function applyNMS(items: NmsItem[], iouThreshold: number): NmsItem[] {
  if (items.length === 0) return [];

  // Sort indices by descending score, then walk the list keeping
  // boxes that don't overlap enough with any already-kept one.
  const sortedIdx = items
    .map((item, idx) => ({ idx, score: item.score }))
    .sort((a, b) => b.score - a.score)
    .map(({ idx }) => idx);

  const kept: number[] = [];
  const suppressed = new Set<number>();

  for (const i of sortedIdx) {
    if (suppressed.has(i)) continue;
    kept.push(i);
    const a = items[i].bbox;
    for (const j of sortedIdx) {
      if (j === i || suppressed.has(j)) continue;
      if (iou(a, items[j].bbox) > iouThreshold) {
        suppressed.add(j);
      }
    }
  }

  return kept.map((idx) => items[idx]);
}

/** Intersection-over-Union for two xyxy boxes. */
export function iou(
  a: { x1: number; y1: number; x2: number; y2: number },
  b: { x1: number; y1: number; x2: number; y2: number },
): number {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter === 0) return 0;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  const union = areaA + areaB - inter;
  return union === 0 ? 0 : inter / union;
}
