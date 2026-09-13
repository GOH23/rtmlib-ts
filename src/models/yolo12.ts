/**
 * YOLO12 object detection — person-only.
 *
 * Compatible with Ultralytics YOLOv12 ONNX export in the standard
 * `[1, num_boxes, 6]` (`x1, y1, x2, y2, score, classId`) layout.
 *
 * Person-only filter (`classId === 0`) is the only difference from
 * `YOLO26`; everything else — preprocess, output parsing, NMS — lives
 * on the shared `YoloBase` (`./_yoloBase`).
 */

import { YoloBase } from './_yoloBase';

export class YOLO12 extends YoloBase {
  /** Keep only COCO class 0 ("person"). */
  protected shouldKeepClass(classId: number): boolean {
    return classId === 0;
  }
}
