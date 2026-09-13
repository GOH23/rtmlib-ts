/**
 * Centralized defaults — magic numbers and named constants shared by the
 * detector classes. Every detector used to inline its own `[640, 640]`
 * / `0.5` / `0.45` / `0.3` literals; this is the single source of truth
 * so a tuning change (e.g. switching the default NMS IoU) happens in
 * one place.
 */

/** YOLO detection input — "speed" preset (416). */
export const YOLO_INPUT_SIZE: readonly [number, number] = [416, 416];

/** YOLO detection input — "accuracy" preset (640). */
export const YOLO_ACCURACY_INPUT_SIZE: readonly [number, number] = [640, 640];

/** RTMW pose model input (H × W). */
export const RTMW_POSE_INPUT_SIZE: readonly [number, number] = [384, 288];

/** ViTPose++ pose model input (H × W). */
export const VITPOSE_INPUT_SIZE: readonly [number, number] = [256, 192];

/** Default detection confidence threshold. */
export const DEFAULT_DET_CONFIDENCE = 0.5;

/** Default pose keypoint confidence threshold. */
export const DEFAULT_POSE_CONFIDENCE = 0.3;

/** Default NMS IoU threshold (standard COCO-style). */
export const DEFAULT_NMS_IOU = 0.45;

/** Tighter NMS IoU used when high-density overlapping detections are
 * expected (e.g. crowd scenes). */
export const STRICT_NMS_IOU = 0.35;

/** Default MediaPipe detector score threshold. */
export const DEFAULT_MP_SCORE_THRESHOLD = 0.5;

/** Default MediaPipe detector max-results. */
export const DEFAULT_MP_MAX_RESULTS = 10;

/** MediaPipe detector input size cap. The detector downscales frames
 * larger than this before inference (matches the Pose3DDetector
 * `mpInputMaxSize` optimization). */
export const MP_INPUT_MAX_SIZE = 640;
