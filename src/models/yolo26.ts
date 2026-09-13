/**
 * YOLO26 object detection — multi-class.
 *
 * Compatible with Ultralytics YOLOv26 ONNX export in the standard
 * `[1, num_boxes, 6]` (`x1, y1, x2, y2, score, classId`) layout.
 *
 * No class filtering — every box above the score threshold survives.
 * Everything else — preprocess, output parsing, NMS — lives on the
 * shared `YoloBase` (`./_yoloBase`).
 */

import { YoloBase } from './_yoloBase';

export class YOLO26 extends YoloBase {
  /** Multi-class: keep everything above the score threshold. */
  protected shouldKeepClass(_classId: number): boolean {
    return true;
  }
}
