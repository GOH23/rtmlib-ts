/**
 * Shared base class for Ultralytics YOLO-family detectors.
 *
 * Both `YOLO12` (person-only) and `YOLO26` (multi-class) consume an
 * ONNX export in the `[1, num_boxes, 6]` layout — `[x1, y1, x2, y2,
 * score, class_id]` per box, no anchor decoding required. They differ
 * only in which `classId`s survive post-processing:
 *
 *   - `YOLO12` keeps `classId === 0` (COCO "person")
 *   - `YOLO26` keeps everything above the score threshold
 *
 * That single difference is what subclasses override via
 * {@link YoloBase.shouldKeepClass}. Everything else — letterbox
 * preprocess (via `core/preprocessing.ts:letterboxCHW`), output
 * parsing, coordinate unpad, and greedy NMS — lives here.
 *
 * Subclassing contract:
 *   1. Override `protected shouldKeepClass(classId: number): boolean`.
 *   2. The constructor signature is identical to `BaseTool` plus the
 *      two YOLO-specific thresholds (`nmsThr`, `scoreThr`), so any
 *      caller doing `new YOLO12(path, size, nms, score)` keeps working
 *      unchanged when the class is replaced by `new YoloBase(...)`.
 */

import { BaseTool } from '../core/base';
import { letterboxCHW } from '../core/preprocessing';
import type { Detection } from '../types/index';

/** Greedy Non-Maximum Suppression over `Detection[]`, sorted by score desc. */
function nmsGreedy(
  detections: Detection[],
  iouThreshold: number,
): Detection[] {
  if (detections.length === 0) return [];

  // Sort by score descending. `sort` mutates, but the caller owns the
  // array — `YOLO12.call()` was already building a fresh `detections`
  // list each invocation, so in-place sort is fine.
  detections.sort((a, b) => b.score - a.score);

  const selected: Detection[] = [];
  const used = new Uint8Array(detections.length);

  for (let i = 0; i < detections.length; i++) {
    if (used[i] !== 0) continue;
    const a = detections[i];
    selected.push(a);
    used[i] = 1;
    const boxA = a.bbox;
    const areaA = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1);

    for (let j = i + 1; j < detections.length; j++) {
      if (used[j] !== 0) continue;
      const boxB = detections[j].bbox;

      const x1 = Math.max(boxA.x1, boxB.x1);
      const y1 = Math.max(boxA.y1, boxB.y1);
      const x2 = Math.min(boxA.x2, boxB.x2);
      const y2 = Math.min(boxA.y2, boxB.y2);
      const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      if (intersection === 0) continue;

      const areaB = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1);
      const union = areaA + areaB - intersection;
      const iou = intersection / union;

      // Suppress when overlap strictly exceeds the threshold. The
      // previous in-class copy used `<=` which inverted the predicate
      // and made NMS a no-op — fixed here.
      if (iou > iouThreshold) used[j] = 1;
    }
  }
  return selected;
}

/**
 * Per-box layout in the ONNX output: `[x1, y1, x2, y2, score, classId]`.
 * Both YOLO12 and YOLO26 emit this exact shape; the only thing that
 * varies is the post-filter.
 */
const BOX_STRIDE = 6;

export abstract class YoloBase extends BaseTool {
  protected readonly nmsThr: number;
  public readonly scoreThr: number;
  private initialized = false;

  constructor(
    modelPath: string,
    modelInputSize: [number, number] = [640, 640],
    nmsThr: number = 0.45,
    scoreThr: number = 0.5,
  ) {
    super(modelPath, modelInputSize);
    this.nmsThr = nmsThr;
    this.scoreThr = scoreThr;
  }

  async init(): Promise<void> {
    await super.init();
    this.initialized = true;
  }

  /**
   * Subclasses filter by class here.
   *  - `YOLO12` returns `classId === 0` (COCO person).
   *  - `YOLO26` returns `true` (multi-class, score-only filter).
   */
  protected abstract shouldKeepClass(classId: number): boolean;

  async call(
    image: Uint8Array,
    imgWidth: number,
    imgHeight: number,
  ): Promise<Detection[]> {
    if (!this.initialized) {
      await this.init();
    }

    const [inputH, inputW] = this.modelInputSize;
    const { tensor, meta } = letterboxCHW(image, imgWidth, imgHeight, inputW, inputH);
    const outputs = await this.inference(tensor);

    const detOutput = outputs[0];
    const detShape = detOutput.dims;
    if (detShape.length !== 3 || detShape[2] !== BOX_STRIDE || detOutput.type !== 'float32') {
      return [];
    }

    const detArray = detOutput.data as Float32Array;
    const numBoxes = detShape[1];
    const detections: Detection[] = [];

    for (let i = 0; i < numBoxes; i++) {
      const baseIdx = i * BOX_STRIDE;
      const x1 = detArray[baseIdx];
      const y1 = detArray[baseIdx + 1];
      const x2 = detArray[baseIdx + 2];
      const y2 = detArray[baseIdx + 3];
      const score = detArray[baseIdx + 4];
      const classId = detArray[baseIdx + 5];

      if (score < this.scoreThr) continue;
      if (!this.shouldKeepClass(classId)) continue;

      const transformedX1 = (x1 - meta.paddingX) * meta.scaleX;
      const transformedY1 = (y1 - meta.paddingY) * meta.scaleY;
      const transformedX2 = (x2 - meta.paddingX) * meta.scaleX;
      const transformedY2 = (y2 - meta.paddingY) * meta.scaleY;
      if (transformedX1 >= transformedX2 || transformedY1 >= transformedY2) continue;

      detections.push({
        bbox: {
          x1: Math.max(0, transformedX1),
          y1: Math.max(0, transformedY1),
          x2: Math.min(imgWidth, transformedX2),
          y2: Math.min(imgHeight, transformedY2),
        },
        score,
        classId: Math.round(classId),
      });
    }

    return nmsGreedy(detections, this.nmsThr);
  }
}
