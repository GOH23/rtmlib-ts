/**
 * Default ONNX URLs for the supported Ultralytics YOLO versions.
 *
 * Each entry points at the same HuggingFace dataset that the rest of the
 * rtmlib-ts demo fetches from. Self-hosting (Vite static /models path
 * with COOP+COEP for multithreaded WASM) only needs to mirror these files
 * under the same filenames; the URL is the only thing detectors see.
 *
 * Version notes:
 * - `yolov8n`  — Ultralytics YOLOv8 nano. Original anchor model. Conservative
 *                 output (`[1, num_boxes, 6]`) and the broadest coverage of
 *                 tested CPU/WebGL/WebGPU paths.
 * - `yolov12n` — YOLO12 nano (attention-augmented). Same output format as
 *                 v8. Currently the default for pose pipelines because of
 *                 good small-person recall.
 * - `yolo26n`  — YOLO26 nano. Newer output shape (`[1, num_boxes, 80+4]`
 *                 — class scores first, cx/cy/w/h last) needs a different
 *                 decoder; ObjectDetector detects this format at runtime.
 */
export type YoloVersion = 'yolov8n' | 'yolov12n' | 'yolo26n';

export const YOLO_VERSIONS: Record<YoloVersion, string> = {
  yolov8n: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov8n.onnx',
  yolov12n: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  yolo26n: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolo26n.onnx',
};

/**
 * Resolve a `YoloVersion` (or passthrough URL) to a concrete model URL.
 * `null` / `undefined` falls back to `yolov12n`, which is the historical
 * default used by PoseDetector and Pose3DDetector.
 */
export function resolveYoloModelUrl(
  input: YoloVersion | string | null | undefined,
): string {
  if (!input) return YOLO_VERSIONS.yolov12n;
  if (input in YOLO_VERSIONS) return YOLO_VERSIONS[input as YoloVersion];
  return input;
}
