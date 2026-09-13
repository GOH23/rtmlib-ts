/**
 * Centralized model URL registry. Every detector used to inline its own
 * HuggingFace / JSDelivr URL literal; this is the single source of truth
 * so a model migration (e.g. switching CDN or bumping a release) happens
 * in one place.
 */

export const HF_RTMLIB_BASE = 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main';

/** MediaPipe EfficientDet-Lite0 (TFLite). */
export const MEDIAPIPE_EFFICIENTDET_URL =
  'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite';

/** MediaPipe TFLite-WASM runtime fileset (vision_wasm_internal.js + .wasm). */
export const MEDIAPIPE_WASM_BASE =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';

/** MediaPipe BlazePose pose-landmarker (`.task` format). */
export const MEDIAPIPE_POSE_LANDMARKER_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

/** Default YOLO12n ONNX URL (RTMLib-style). */
export const YOLOV12N_MODEL_URL = `${HF_RTMLIB_BASE}/yolov12n.onnx`;
