/**
 * Pose3DDetector — unified 3D-pose estimation pipeline.
 *
 * Eight pipelines, one class. The pipeline is composed from two orthogonal
 * **required** config fields — `objectModel` (which person detector) and
 * `pose3dModel` (which 3D pose model):
 *
 *   - `objectModel: 'yolov{8,12,26}n' | 'mediapipe'` (default `'yolov12n'`)
 *     picks the person detector. The three `yolo*` values differ in which
 *     Ultralytics YOLO export runs; `'mediapipe'` uses EfficientDet-Lite0.
 *   - `pose3dModel: 'rtmw3d' | 'instanthmr'` (default `'rtmw3d'`)
 *     picks the 3D pose model. `'rtmw3d'` returns 17-keypoint COCO17 3D
 *     pose in metres; `'instanthmr'` returns the 70-keypoint MHR mesh
 *     (body, hands, face) in metres, plus MHR pose parameters, body
 *     shape, and camera translation. WebGL is incompatible with InstantHMR
 *     (use wasm / webgpu / webnn).
 *   - `backend: 'wasm' | 'webgl' | 'webgpu' | 'webnn'` (required) — the
 *     ONNX execution provider.
 *
 * All ONNX. There is no TFLite/LiteRT code path.
 *
 * @example
 * ```ts
 * // YOLOv12 + RTMW3D
 * const a = new Pose3DDetector({
 *   objectModel: 'yolov12n',
 *   pose3dModel: 'rtmw3d',
 *   backend: 'webgl',
 * });
 * await a.init();
 * const r = await a.detectFromCanvas(canvas);
 * console.log(r.keypoints[0][0]); // [x, y, z] of the first person, first keypoint
 *
 * // MediaPipe + RTMW3D (faster on large frames)
 * const b = new Pose3DDetector({
 *   objectModel: 'mediapipe',
 *   pose3dModel: 'rtmw3d',
 *   backend: 'wasm',
 *   profile: true,
 * });
 * await b.init();
 * const r2 = await b.detectFromCanvas(canvas);
 * console.log(r2.stats); // { personCount, detTime, poseTime, totalTime }
 *
 * // YOLOv26n + InstantHMR (MHR70 3D mesh)
 * const c = new Pose3DDetector({
 *   objectModel: 'yolo26n',
 *   pose3dModel: 'instanthmr',
 *   backend: 'wasm',
 * });
 * await c.init();
 * const r3 = await c.detectFromCanvas(canvas);
 * console.log(r3.persons[0].keypoints3d[5].name); // 'left_shoulder'
 * ```
 */

import * as ort from 'onnxruntime-web/all';

import { FilesetResolver, ObjectDetector as MPObjectDetector } from '@mediapipe/tasks-vision';

import { BaseTool } from '../core/base';
import { initOnnxRuntimeWeb } from '../core/onnxRuntime';
import { loadOnnxSession } from '../core/onnxSession';
import { fillLetterbox, letterboxGeometry, rgbaToCHW } from '../core/preprocessing';
import { loadBitmapFromBlob, loadImageFromFile } from '../core/sourceLoaders';
import { createLogger } from '../core/logger';

import { InstantHMRModel, INSTANTHMR_MODEL_URL } from '../models/instanthmr';
import { resolveYoloModelUrl, type YoloVersion } from '../models/yoloModels';
import {
  INPUT_SIZE as INSTANTHMR_INPUT_SIZE,
  NUM_JOINTS as INSTANTHMR_NUM_JOINTS,
  JOINT_NAMES as INSTANTHMR_JOINT_NAMES,
  type BBox as InstHmrBBox,
  type CropBox,
  cropBoxFor,
  cliffCondFor,
  pixelsToNCHW,
  denormalizeJoints2D,
} from '../core/instanthmrGeometry';
import type { WebNNProviderOptionsOrUndefined } from '../types/index';

// Configure ONNX Runtime Web at module top level (no-op under SSR).
initOnnxRuntimeWeb();

const log = createLogger('Pose3DDetector');

// =============================================================================
// Public types
// =============================================================================

/** Person detector variants accepted by `Pose3DDetectorConfig.objectModel`. */
export type Pose3DObjectModel = 'yolov8n' | 'yolov12n' | 'yolo26n' | 'mediapipe';

/** 3D pose model variants accepted by `Pose3DDetectorConfig.pose3dModel`. */
export type Pose3DModel = 'rtmw3d' | 'instanthmr';

/** Result type of `Pose3DDetector.detectFrom*()` when
 * `pose3dModel === 'rtmw3d'`.
 *
 * Keypoint count: the bundled RTMW3D-X `cocktail14` export emits
 * **133 keypoints per person** (COCO-WholeBody layout: body 17 + face
 * 68 + hands 21×2 + feet 6). The shapes below are parameterised by K
 * (= model's output channels, read from `shx[1]` at runtime, default
 * 133 for the bundled model). For a COCO17-only layout, slice
 * `keypoints[i].slice(0, 17)` etc. — index 0..16 is the body subset.
 *
 * Skeleton presets for both layouts are exported as `coco17` (17 body
 * keypoints) and `coco133` (full 133-key COCO-WholeBody layout).
 */
export interface Pose3DResult {
  /** Per-person, per-keypoint 3D coordinates in metres `[x, y, z]`. Shape `[N][K][3]`. */
  keypoints: number[][][];
  /** Per-keypoint score in `[0, 1]` (max of X/Y simcc peaks). Shape `[N][K]`. */
  scores: number[][];
  /** Per-keypoint normalised SimCC peak `[normX, normY, normZ]` in `[0, 1]`. Shape `[N][K][3]`. */
  keypointsSimcc: number[][][];
  /** Per-keypoint 2D pixel coords in the source frame. Shape `[N][K][2]`. */
  keypoints2d: number[][][];
  /** Timing breakdown — always populated; mpMs/preprocessMs/inferMs/postprocessMs only when `profile: true`. */
  stats?: Pose3DStats;
}

/** Per-call timing breakdown. `mpMs` etc. only populated when `config.profile === true`. */
export interface Pose3DStats {
  personCount: number;
  detTime: number;
  poseTime: number;
  totalTime: number;
  /** Person-detector time in ms (`objectModel === 'mediapipe'` or
   * `'yolo*'`; reused as the detector slot for whichever person detector
   * ran). Undefined when the pipeline has no detector. */
  mpMs?: number;
  preprocessMs?: number;
  inferMs?: number;
  postprocessMs?: number;
}

/** A single person detected by an rtmw3d pipeline. */
export interface Person3D {
  bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number };
  keypoints: number[][];
  scores: number[];
  keypoints2d: number[][];
  keypointsSimcc: number[][];
}

/** Per-person 70-keypoint 3D pose from the InstantHMR pipeline (instanthmr only). */
export interface InstantHMRPerson {
  bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number };
  /** 70 3D keypoints in metres, camera frame. */
  keypoints3d: InstantHMRKeypoint3D[];
  /** 70 2D keypoints in source-frame pixels. */
  keypoints2d: Array<{ x: number; y: number; id: number; name: string }>;
  /** MHR pose parameters (length 204: 34 joints × 6D rotation). */
  mhr: Float32Array;
  /** Body + head + hand shape parameters (length 45). */
  shape: Float32Array;
  /** Camera translation in metres, length 3. */
  cam: [number, number, number];
  /** Raw `joints_2d` from the model in normalised crop coords `[-1, 1]`.
   * Length 70 × 2. Only populated when the detector was constructed with
   * `debugJointOffsets: true` — used for diagnosing 2D-overlay alignment. */
  joints2dNorm?: number[][];
  /** Crop box used to feed the InstantHMR model. Needed to back-project
   * `joints2dNorm` into source-frame pixels. Only populated when
   * `debugJointOffsets: true`. */
  crop?: { x0: number; y0: number; size: number };
}

/** A single 3D keypoint in metres (instanthmr only). */
export interface InstantHMRKeypoint3D {
  x: number;
  y: number;
  z: number;
  score?: number;
  id: number;
  name: string;
}

/** Result type for the InstantHMR pipeline. */
export interface InstantHMR3DResult {
  persons: InstantHMRPerson[];
  stats?: Pose3DStats & { otherMs?: number };
}

/** Per-stage timing breakdown, exposed via `detector.lastProfile` when `profile: true`. */
export interface Pose3DProfile {
  mpMs: number;
  preprocessMs: number;
  inferMs: number;
  postprocessMs: number;
  otherMs: number;
  personCount: number;
  totalMs: number;
}

/** Configuration for `Pose3DDetector`. The pipeline is composed from three
 * orthogonal **required** selectors — `objectModel` (which person detector),
 * `pose3dModel` (which 3D pose model), and `backend` (which ONNX execution
 * provider). Every `objectModel × pose3dModel` combination is expressible
 * (8 total). Fields with a `mediaPipe*` / `mp*` prefix are silently ignored
 * unless the chosen `objectModel === 'mediapipe'`; `bboxExpansion` and
 * `detectorStride` are only consulted when `pose3dModel === 'instanthmr'`. */
export interface Pose3DDetectorConfig {
  // ---- Pipeline selectors (orthogonal, all required) ----
  /** Person detector. `'yolo*'` resolves to a HuggingFace URL via
   * `YOLO_VERSIONS[version]`; `'mediapipe'` uses EfficientDet-Lite0.
   * `'yolo*'` is roughly an order of magnitude faster on most frames
   * (YOLOv12n is ~10 MB / ~10 ms on multithreaded WASM vs MediaPipe's
   * ~60 ms). `'mediapipe'` is the right choice on very large frames where
   * `mpInputMaxSize` downscale caps its cost. */
  objectModel: Pose3DObjectModel;

  /** 3D pose model. `'rtmw3d'` returns COCO17 3D keypoints;
   * `'instanthmr'` returns the MHR70 mesh + camera translation +
   * body shape. */
  pose3dModel: Pose3DModel;

  /** Execution provider for the pose (and YOLO) ONNX session. The
   * InstantHMR graph is incompatible with WebGL — pick `'wasm'` /
   * `'webgpu'` / `'webnn'` when `pose3dModel === 'instanthmr'`. */
  backend: 'wasm' | 'webgl' | 'webgpu' | 'webnn';

  // ---- Raw URL overrides (power users / self-hosting / COEP) ----
  /** Person detector ONNX model URL. When `objectModel` is a YOLO variant
   * this wins over `objectModel`; otherwise it defaults to
   * `YOLO_VERSIONS[objectModel]` (or the MediaPipe EfficientDet URL).
   * Cache hits serve the bytes same-origin so this URL doesn't have to be
   * CORP-sending — see `getCachedModel()`. */
  detModel?: string;
  /** RTMW3D pose ONNX URL (only used when `pose3dModel === 'rtmw3d'`).
   * Default: HuggingFace RTMW3D-X ONNX. */
  poseModel?: string;
  /** MediaPipe EfficientDet-Lite0 URL (only used when `objectModel ===
   * 'mediapipe'`). Default: Google Storage. */
  mediaPipeModelPath?: string;

  // ---- Detection tunables (apply to whichever detector objectModel picks) ----
  /** Detection input size as `[width, height]`. Default `[640, 640]`. */
  detInputSize?: [number, number];
  /** Detection confidence threshold. Default `0.45`. */
  detConfidence?: number;
  /** NMS IoU threshold. Default `0.7`. */
  nmsThreshold?: number;
  /** MediaPipe score threshold. Default `0.5`. Ignored unless `objectModel
   * === 'mediapipe'`. */
  mediaPipeScoreThreshold?: number;
  /** MediaPipe max results, `-1` for all. Default `-1`. Ignored unless
   * `objectModel === 'mediapipe'`. */
  mediaPipeMaxResults?: number;
  /** Downscale source for MediaPipe detect (longest edge, px). Default
   * `640`. Set `0` to disable. Ignored unless `objectModel === 'mediapipe'`. */
  mpInputMaxSize?: number;
  /** Restrict MediaPipe to the `person` category. Default `true`. Ignored
   * unless `objectModel === 'mediapipe'`. */
  personsOnly?: boolean;

  // ---- Pose tunables ----
  /** RTMW3D pose input as `[width, height]`. Default `[288, 384]`. Ignored
   * when `pose3dModel === 'instanthmr'`. */
  poseInputSize?: [number, number];
  /** RTMW3D pose keypoint confidence. Default `0.3`. Ignored when
   * `pose3dModel === 'instanthmr'`. */
  poseConfidence?: number;
  /** Z-axis range in metres for RTMW3D output. Default `2.1744869`. Ignored
   * when `pose3dModel === 'instanthmr'`. */
  zRange?: number;
  /** Square crop expansion for InstantHMR. Default `1.2`. Only consulted
   * when `pose3dModel === 'instanthmr'`. */
  bboxExpansion?: number;

  // ---- instanthmr-only: detector stride ----
  /** Person detector runs every Nth frame; on the skipped frames the last
   * detected bboxes are reused (slightly expanded to absorb motion). The
   * pose model still runs on every frame. Default `1` = detect every
   * frame. Use `2`–`3` on video where the detector dominates per-frame
   * cost (single-threaded WASM, mobile). Only consulted when `pose3dModel
   * === 'instanthmr'`. */
  detectorStride?: number;

  // ---- ONNX runtime knobs (all combinations) ----
  deviceType?: 'cpu' | 'gpu' | 'npu';
  powerPreference?: 'default' | 'low-power' | 'high-performance';
  webnnOptions?: WebNNProviderOptionsOrUndefined;

  // ---- Shared ----
  /** Cache model weights in the Cache API (ONNX) or IndexedDB (MediaPipe
   * `.tflite`). Default `true`. Combined with cache-first model loading
   * (`getCachedModel()`), this is what makes the COEP + HuggingFace
   * combination work — warm-cache responses are served same-origin and
   * survive `require-corp`. */
  cache?: boolean;
  /** Populate `detector.lastProfile` per call. Default `false`. */
  profile?: boolean;
  /** Init progress callback. Stage tokens: `'start'`, `'mp-init'`, `'mp-ready'`,
   * `'yolo-load'`, `'pose-load'`, `'pose-ready'`, `'ready'`. */
  onInitProgress?: (stage: string, detail?: string) => void;
  /** Populate per-person `joints2dNorm` (raw `[-1,1]` crop coords from the
   * model) and `crop` on `InstantHMR3DResult.persons`. Diagnostic only —
   * for debugging 2D-overlay misalignment. Default `false`. */
  debugJointOffsets?: boolean;
}

// =============================================================================
// Pipeline-internal constants
// =============================================================================

const POSE_MODEL_RTMW3D_DEFAULT =
  'https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx';

const YOLO_MODEL_DEFAULT =
  'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx';

const MEDIAPIPE_EFFICIENTDET_URL =
  'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite';

const MEDIAPIPE_WASM_BASE =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';

const PERSON_CATEGORY = 'person';

const RTMW3D_DEFAULT_CONFIG = {
  detInputSize: [640, 640] as [number, number],
  detConfidence: 0.45,
  nmsThreshold: 0.7,
  poseInputSize: [288, 384] as [number, number],
  poseConfidence: 0.3,
  zRange: 2.1744869,
  backend: 'webgl' as 'wasm' | 'webgl' | 'webgpu' | 'webnn',
} as const;

const MEDIAPIPE_DEFAULT_CONFIG = {
  mediaPipeScoreThreshold: 0.5,
  mediaPipeMaxResults: -1,
  mpInputMaxSize: 640,
  personsOnly: true,
  backend: 'wasm' as 'wasm' | 'webgl' | 'webgpu' | 'webnn',
} as const;

const INSTANTHMR_DEFAULT_CONFIG = {
  poseModel: INSTANTHMR_MODEL_URL,
  backend: 'wasm' as 'wasm' | 'webgl' | 'webgpu' | 'webnn',
  bboxExpansion: 1.2,
  mediaPipeScoreThreshold: 0.5,
  mediaPipeMaxResults: -1,
  mpInputMaxSize: 640,
  personsOnly: true,
  // 1 = detect every frame. On 1080p video where YOLO is the dominant cost,
  // stride=2–3 cuts the detector budget in half or third with minor bbox
  // drift (InstantHMR's 1.2× bbox-expansion crop is forgiving).
  detectorStride: 1,
} as const;

const SHARED_DEFAULT_CONFIG = {
  cache: true,
  profile: false,
  deviceType: 'gpu' as 'cpu' | 'gpu' | 'npu',
  powerPreference: 'high-performance' as 'default' | 'low-power' | 'high-performance',
  webnnOptions: undefined as WebNNProviderOptionsOrUndefined,
  onInitProgress: () => { /* noop */ },
} as const;

// =============================================================================
// TS-level: result type picker based on the generic config
// =============================================================================

/** Subset of `Pose3DDetectorConfig` that controls the TS-level return type.
 * Used as the generic parameter on the class — `new Pose3DDetector<{ pose3dModel:
 * 'instanthmr' }>()` returns `InstantHMR3DResult` from `detectFromCanvas`,
 * `'rtmw3d'` returns `Pose3DResult`. */
export interface Pose3DDetectorConfigGeneric {
  pose3dModel: Pose3DModel;
}

/** Result type of `detectFrom*` given the generic config. TypeScript only. */
export type Pose3DDetectorResult<T extends Pose3DDetectorConfigGeneric> =
  T extends { pose3dModel: 'instanthmr' } ? InstantHMR3DResult : Pose3DResult;

// =============================================================================
// Private pipeline subcomponents
// =============================================================================

/** Plain bounding box used between subcomponents. */
interface BBoxRect {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
}

// ---- YOLO detector (per YOLO12 spec: each detection row is 6 floats ---------
// [x1, y1, x2, y2, conf, class_id]). Person-only filter at classId===0 is the
// original pose3dDetector behavior — matches YOLO12 hardcoding.

class YoloDetector {
  private session: ort.InferenceSession | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  readonly inputW: number;
  readonly inputH: number;
  readonly detConfidence: number;
  readonly nmsThreshold: number;
  readonly backend: 'wasm' | 'webgl' | 'webgpu' | 'webnn';

  constructor(cfg: {
    detInputSize: [number, number];
    detConfidence: number;
    nmsThreshold: number;
    backend: 'wasm' | 'webgl' | 'webgpu' | 'webnn';
  }) {
    [this.inputW, this.inputH] = cfg.detInputSize;
    this.detConfidence = cfg.detConfidence;
    this.nmsThreshold = cfg.nmsThreshold;
    this.backend = cfg.backend;
  }

  async init(detModel: string, cache: boolean): Promise<void> {
    this.session = await loadOnnxSession({
      url: detModel,
      backend: this.backend,
      cache,
      logPrefix: '[Pose3DDetector] Detection',
    });

    this.canvas = document.createElement('canvas');
    this.canvas.width = this.inputW;
    this.canvas.height = this.inputH;
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true, alpha: false })!;
  }

  async detect(rgba: Uint8Array, iw: number, ih: number): Promise<BBoxRect[]> {
    if (!this.session) throw new Error('YoloDetector not initialised');
    const [tw, th] = [this.inputW, this.inputH];

    const ctx = this.ctx!;
    const geom = letterboxGeometry(iw, ih, tw, th);
    fillLetterbox(ctx, tw, th);
    const src = document.createElement('canvas');
    const srcCtx = src.getContext('2d')!;
    src.width = iw;
    src.height = ih;
    const srcImg = srcCtx.createImageData(iw, ih);
    srcImg.data.set(rgba);
    srcCtx.putImageData(srcImg, 0, 0);
    ctx.drawImage(src, 0, 0, iw, ih, geom.offX, geom.offY, geom.drawW, geom.drawH);

    const padded = ctx.getImageData(0, 0, tw, th);
    const tensor = new Float32Array(tw * th * 3);
    rgbaToCHW(padded.data, tw, th, tensor);

    const inp = new ort.Tensor('float32', tensor, [1, 3, th, tw]);
    const feeds: Record<string, ort.Tensor> = {};
    const inputName = this.session.inputNames[0];
    feeds[inputName] = inp;
    const results = await this.session.run(feeds);
    const out = results[this.session.outputNames[0]];

    return this.decodeOutput(
      out.data as Float32Array,
      out.dims as readonly number[],
      iw, ih, geom.offX, geom.offY, geom.scaleX, geom.scaleY,
    );
  }

  /** Canvas-direct path — skips the 8 MB RGBA round-trip that `detect(rgba)`
   * does for high-resolution video frames. The source canvas is drawn onto
   * the padded 640×640 canvas via GPU `drawImage`, then only the 1.6 MB
   * padded pixels are read with `getImageData`. On 1080p video this saves
   * ~80-200 ms of pure JS pixel-copy cost vs. `detect(rgba)`. */
  async detectCanvas(src: HTMLCanvasElement): Promise<BBoxRect[]> {
    if (!this.session) throw new Error('YoloDetector not initialised');
    const [tw, th] = [this.inputW, this.inputH];
    const iw = src.width;
    const ih = src.height;

    const ctx = this.ctx!;
    const geom = letterboxGeometry(iw, ih, tw, th);
    fillLetterbox(ctx, tw, th);
    ctx.drawImage(src, 0, 0, iw, ih, geom.offX, geom.offY, geom.drawW, geom.drawH);

    const padded = ctx.getImageData(0, 0, tw, th);
    const tensor = new Float32Array(tw * th * 3);
    rgbaToCHW(padded.data, tw, th, tensor);

    const inp = new ort.Tensor('float32', tensor, [1, 3, th, tw]);
    const feeds: Record<string, ort.Tensor> = {};
    const inputName = this.session.inputNames[0];
    feeds[inputName] = inp;
    const results = await this.session.run(feeds);
    const out = results[this.session.outputNames[0]];

    return this.decodeOutput(
      out.data as Float32Array,
      out.dims as readonly number[],
      iw, ih, geom.offX, geom.offY, geom.scaleX, geom.scaleY,
    );
  }

  private postprocess(
    output: Float32Array,
    numDet: number,
    iw: number,
    ih: number,
    padX: number,
    padY: number,
    scaleX: number,
    scaleY: number,
  ): BBoxRect[] {
    const detected: BBoxRect[] = [];
    for (let i = 0; i < numDet; i++) {
      const idx = i * 6;
      const x1 = output[idx];
      const y1 = output[idx + 1];
      const x2 = output[idx + 2];
      const y2 = output[idx + 3];
      const conf = output[idx + 4];
      const cls = Math.round(output[idx + 5]);
      if (conf < this.detConfidence || cls !== 0) continue;
      const tx1 = (x1 - padX) * scaleX;
      const ty1 = (y1 - padY) * scaleY;
      const tx2 = (x2 - padX) * scaleX;
      const ty2 = (y2 - padY) * scaleY;
      detected.push({
        x1: Math.max(0, tx1),
        y1: Math.max(0, ty1),
        x2: Math.min(iw, tx2),
        y2: Math.min(ih, ty2),
        confidence: conf,
      });
    }
    return applyNMS(detected, this.nmsThreshold);
  }

  /**
   * Decode raw Ultralytics YOLOv8/YOLOv11-style output (no end2end):
   * `[batch, 4 + num_classes, num_anchors]`. Per anchor i:
   *   channels[0]      = bbox cx (input-image pixels)
   *   channels[1]      = bbox cy
   *   channels[2]      = bbox w
   *   channels[3]      = bbox h
   *   channels[4..end] = per-class scores (already sigmoid-ed → [0, 1])
   *
   * ONNX Runtime Web returns the underlying `Float32Array` in row-major
   * (C-order) layout, so for a channels-first tensor `[B, C, A]` the offset
   * for value at channel c, anchor i is `c * A + i` — NOT `i * C + c`.
   *
   * Returns the same `BBoxRect[]` shape `postprocess()` produces, so the
   * rest of `YoloDetector.detect*()` is identical to the end2end path.
   */
  private postprocessRaw(
    output: Float32Array,
    numAnchors: number,
    numChannels: number,
    numClasses: number,
    iw: number,
    ih: number,
    padX: number,
    padY: number,
    scaleX: number,
    scaleY: number,
  ): BBoxRect[] {
    const detected: BBoxRect[] = [];
    // Strides for C-order channels-first `[B, numChannels, numAnchors]`:
    //   channel c at anchor i → output[c * numAnchors + i]
    const cxOff  = 0 * numAnchors;
    const cyOff  = 1 * numAnchors;
    const wOff   = 2 * numAnchors;
    const hOff   = 3 * numAnchors;
    const clsOff = 4 * numAnchors; // class 0 (person) stride; only one we read
    // Hardcoded: COCO class 0 = person.
    for (let i = 0; i < numAnchors; i++) {
      const score = output[clsOff + i];
      if (score < this.detConfidence) continue;
      const cx = output[cxOff + i];
      const cy = output[cyOff + i];
      const w  = output[wOff  + i];
      const h  = output[hOff  + i];
      let x1 = cx - w * 0.5;
      let y1 = cy - h * 0.5;
      let x2 = cx + w * 0.5;
      let y2 = cy + h * 0.5;
      // Sanity: a zero-sized box means the anchor never matched anything.
      if (x2 - x1 <= 1 || y2 - y1 <= 1) continue;
      x1 = (x1 - padX) * scaleX;
      y1 = (y1 - padY) * scaleY;
      x2 = (x2 - padX) * scaleX;
      y2 = (y2 - padY) * scaleY;
      detected.push({
        x1: Math.max(0, x1),
        y1: Math.max(0, y1),
        x2: Math.min(iw, x2),
        y2: Math.min(ih, y2),
        confidence: score,
      });
    }
    return applyNMS(detected, this.nmsThreshold);
  }

  /**
   * Inspect the ONNX output dims and dispatch to the right decoder:
   *   - `[1, num_boxes, 6]`   → end2end-processed (`x1, y1, x2, y2, conf, cls`)
   *   - `[1, 4 + num_classes, num_anchors]` → raw Ultralytics (`cx, cy, w, h, cls0, …`)
   *
   * yolov8n ships raw, yolov12n + yolo26n ship end2end — see
   * `scripts/test-yolo-shapes.py` for the per-model dims.
   */
  private decodeOutput(
    output: Float32Array,
    dims: readonly number[],
    iw: number,
    ih: number,
    padX: number,
    padY: number,
    scaleX: number,
    scaleY: number,
  ): BBoxRect[] {
    // End2end: [batch, num_boxes, 6].
    if (dims.length === 3 && dims[2] === 6) {
      return this.postprocess(output, dims[1], iw, ih, padX, padY, scaleX, scaleY);
    }
    // Raw: [batch, 4 + num_classes, num_anchors], channels-first.
    if (dims.length === 3 && dims[1] >= 5 && dims[2] > dims[1]) {
      const numClasses = dims[1] - 4;
      return this.postprocessRaw(output, dims[2], dims[1], numClasses, iw, ih, padX, padY, scaleX, scaleY);
    }
    log.warn(`[YoloDetector] Unexpected YOLO output dims [${dims.join(',')}]; returning no boxes`);
    return [];
  }

  dispose(): void {
    if (this.session) { this.session.release(); this.session = null; }
  }
}

// ---- MediaPipe person detector (FilesetResolver + EfficientDet-Lite0) -------

class MediaPipePersonDetector {
  private mp: MPObjectDetector | null = null;
  private vision: any = null;
  private mpInput: HTMLCanvasElement | null = null;
  private mpInputCtx: CanvasRenderingContext2D | null = null;
  private mpInputSrcW = 0;
  private mpInputSrcH = 0;
  private mpInputDstW = 0;
  private mpInputDstH = 0;
  readonly mpScoreThreshold: number;
  readonly mpMaxResults: number;
  readonly personsOnly: boolean;
  readonly mpInputMaxSize: number;

  constructor(cfg: {
    mediaPipeScoreThreshold: number;
    mediaPipeMaxResults: number;
    personsOnly: boolean;
    mpInputMaxSize: number;
  }) {
    this.mpScoreThreshold = cfg.mediaPipeScoreThreshold;
    this.mpMaxResults = cfg.mediaPipeMaxResults;
    this.personsOnly = cfg.personsOnly;
    this.mpInputMaxSize = cfg.mpInputMaxSize;
  }

  async init(modelPath: string): Promise<void> {
    this.vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_BASE);
    this.mp = await MPObjectDetector.createFromOptions(this.vision, {
      baseOptions: { modelAssetPath: modelPath },
      scoreThreshold: this.mpScoreThreshold,
      maxResults: this.mpMaxResults,
      categoryAllowlist: this.personsOnly ? [PERSON_CATEGORY] : undefined,
      runningMode: 'IMAGE',
    });
  }

  /** Returns the time spent inside MediaPipe `detect` (for profiling). */
  async detect(canvas: HTMLCanvasElement): Promise<{ boxes: BBoxRect[]; mpMs: number }> {
    if (!this.mp) throw new Error('MediaPipePersonDetector not initialised');
    const [scaled, invScale] = this.getScaledCanvas(canvas);
    const t0 = performance.now();
    const res = this.mp.detect(scaled);
    const t1 = performance.now();
    const boxes: BBoxRect[] = [];
    for (const d of res.detections) {
      if (this.personsOnly && d.categories[0]?.categoryName?.toLowerCase() !== PERSON_CATEGORY) continue;
      const ob = d.boundingBox;
      if (!ob || !(ob.width > 0) || !(ob.height > 0)) continue;
      const x1 = ob.originX * invScale;
      const y1 = ob.originY * invScale;
      const w = ob.width * invScale;
      const h = ob.height * invScale;
      if (w < 2 || h < 2) continue;
      boxes.push({
        x1,
        y1,
        x2: x1 + w,
        y2: y1 + h,
        confidence: d.categories[0]?.score ?? 0,
      });
    }
    return { boxes, mpMs: t1 - t0 };
  }

  /** Returns a canvas no larger than `mpInputMaxSize` on its longest edge,
   * reusing the same backing canvas across calls when the source/target size
   * class doesn't change. The second return value is the inverse scale the
   * caller must apply to bbox coords. */
  private getScaledCanvas(src: HTMLCanvasElement): [HTMLCanvasElement, number] {
    const max = this.mpInputMaxSize;
    if (!max || max <= 0 || (src.width <= max && src.height <= max)) {
      return [src, 1];
    }
    const s = Math.min(max / src.width, max / src.height);
    const sw = Math.max(1, Math.round(src.width * s));
    const sh = Math.max(1, Math.round(src.height * s));
    if (!this.mpInput ||
        this.mpInputSrcW !== src.width || this.mpInputSrcH !== src.height ||
        this.mpInputDstW !== sw || this.mpInputDstH !== sh) {
      this.mpInput = document.createElement('canvas');
      this.mpInput.width = sw;
      this.mpInput.height = sh;
      this.mpInputCtx = this.mpInput.getContext('2d', { alpha: false })!;
      this.mpInputSrcW = src.width;
      this.mpInputSrcH = src.height;
      this.mpInputDstW = sw;
      this.mpInputDstH = sh;
    }
    this.mpInputCtx!.drawImage(src, 0, 0, sw, sh);
    return [this.mpInput!, 1 / s];
  }

  async setScoreThreshold(t: number): Promise<void> {
    if (!this.mp) return;
    await this.mp.setOptions({ scoreThreshold: t });
  }

  dispose(): void {
    if (this.mp) {
      try { this.mp.close(); } catch { /* swallow */ }
      this.mp = null;
    }
  }
}

// ---- RTMW3D pose: ONNX inference session + SimCC decode ---------------------

interface Pose3DInference {
  keypoints: number[][];
  scores: number[];
  keypoints2d: number[][];
  keypointsSimcc: number[][];
  preprocessMs: number;
  inferMs: number;
  postprocessMs: number;
}

class Rtmw3DPose {
  private session: ort.InferenceSession | null = null;
  private cachedInputName = '';
  private cachedFeeds: Record<string, ort.Tensor> = {};
  private cachedSimccX = '';
  private cachedSimccY = '';
  private cachedSimccZ = '';
  private poseCanvas: HTMLCanvasElement | null = null;
  private poseCtx: CanvasRenderingContext2D | null = null;
  private poseTensorBuffer: Float32Array | null = null;
  private srcPoseCanvas: HTMLCanvasElement | null = null;
  private srcPoseCtx: CanvasRenderingContext2D | null = null;
  private srcPoseW = 0;
  private srcPoseH = 0;
  readonly inputW: number;
  readonly inputH: number;
  readonly poseConfidence: number;
  readonly zRange: number;

  constructor(cfg: {
    poseInputSize: [number, number];
    poseConfidence: number;
    zRange: number;
  }) {
    [this.inputW, this.inputH] = cfg.poseInputSize;
    this.poseConfidence = cfg.poseConfidence;
    this.zRange = cfg.zRange;
  }

  async init(poseModel: string, backend: 'wasm' | 'webgl' | 'webgpu' | 'webnn', cache: boolean): Promise<void> {
    this.session = await loadOnnxSession({
      url: poseModel,
      backend,
      cache,
      logPrefix: '[Pose3DDetector] 3D Pose',
    });

    // Probe to discover SimCC X/Y/Z output names by shape.
    const outputNames = this.session.outputNames;
    const probeFeeds: Record<string, ort.Tensor> = {};
    probeFeeds[this.session.inputNames[0]] = new ort.Tensor(
      'float32', new Float32Array(this.inputW * this.inputH * 3),
      [1, 3, this.inputH, this.inputW],
    );
    const probe = await this.session.run(probeFeeds);
    const d0 = (probe[outputNames[0]].dims as number[])[2];
    const d1 = (probe[outputNames[1]].dims as number[])[2];
    const d2 = (probe[outputNames[2]].dims as number[])[2];
    const xIdx = [d0, d1, d2].indexOf(576);
    const yIdx = [d0, d1, d2].indexOf(768);
    const zIdx = 3 - xIdx - yIdx;
    this.cachedInputName = this.session.inputNames[0];
    this.cachedSimccX = outputNames[xIdx];
    this.cachedSimccY = outputNames[yIdx];
    this.cachedSimccZ = outputNames[zIdx];

    this.poseCanvas = document.createElement('canvas');
    this.poseCanvas.width = this.inputW;
    this.poseCanvas.height = this.inputH;
    this.poseCtx = this.poseCanvas.getContext('2d', { willReadFrequently: true, alpha: false })!;
    this.poseTensorBuffer = new Float32Array(3 * this.inputW * this.inputH);
  }

  async infer(canvas: HTMLCanvasElement, box: BBoxRect): Promise<Pose3DInference> {
    if (!this.session) throw new Error('Rtmw3DPose not initialised');

    const [iw, ih] = [canvas.width, canvas.height];

    const tPre0 = performance.now();
    const { tensor, center, scale } = this.preprocess(canvas, iw, ih, box);
    const tPre1 = performance.now();

    this.cachedFeeds[this.cachedInputName] = new ort.Tensor(
      'float32', tensor, [1, 3, this.inputH, this.inputW],
    );
    const tInf0 = performance.now();
    const out = await this.session.run(this.cachedFeeds);
    const tInf1 = performance.now();

    const x = out[this.cachedSimccX];
    const y = out[this.cachedSimccY];
    const z = out[this.cachedSimccZ];
    const tPost0 = performance.now();
    const res = decodeSimCC(
      x.data as Float32Array, y.data as Float32Array, z.data as Float32Array,
      x.dims as number[], y.dims as number[], z.dims as number[],
      center, scale, iw, ih, this.poseConfidence, this.zRange,
    );
    const tPost1 = performance.now();

    return {
      ...res,
      preprocessMs: tPre1 - tPre0,
      inferMs: tInf1 - tInf0,
      postprocessMs: tPost1 - tPost0,
    };
  }

  /** Affine crop + ImageNet normalization into a NCHW float32 tensor. */
  private preprocess(
    canvas: HTMLCanvasElement,
    iw: number, ih: number,
    bbox: BBoxRect,
  ): { tensor: Float32Array; center: [number, number]; scale: [number, number] } {
    const [tw, th] = [this.inputW, this.inputH];
    const bw = bbox.x2 - bbox.x1;
    const bh = bbox.y2 - bbox.y1;
    const center: [number, number] = [bbox.x1 + bw / 2, bbox.y1 + bh / 2];

    // padding 1.25 (matches rtmlib reference).
    let scaleW = bw * 1.25;
    let scaleH = bh * 1.25;
    const modelAR = tw / th;
    if (scaleW / scaleH > modelAR) scaleH = scaleW / modelAR;
    else scaleW = scaleH * modelAR;
    const scale: [number, number] = [scaleW, scaleH];

    // Reuse a single source canvas per source size class — recreating a
    // 2D context per call would dominate runtime.
    if (!this.srcPoseCanvas ||
        this.srcPoseW !== iw || this.srcPoseH !== ih ||
        this.srcPoseCanvas !== canvas) {
      this.srcPoseCanvas = canvas;
      this.srcPoseCtx = canvas.getContext('2d', { willReadFrequently: true, alpha: false })!;
      this.srcPoseW = iw;
      this.srcPoseH = ih;
    }

    const ctx = this.poseCtx!;
    ctx.clearRect(0, 0, tw, th);
    const srcX = center[0] - scaleW / 2;
    const srcY = center[1] - scaleH / 2;
    ctx.drawImage(this.srcPoseCanvas, srcX, srcY, scaleW, scaleH, 0, 0, tw, th);

    const cropped = ctx.getImageData(0, 0, tw, th);
    const tensor = this.poseTensorBuffer!;
    const len = cropped.data.length;
    const plane = tw * th;
    const mean0 = 123.675, mean1 = 116.28, mean2 = 103.53;
    const inv0 = 1 / 58.395, inv1 = 1 / 57.12, inv2 = 1 / 57.375;
    for (let i = 0; i < len; i += 16) {
      const p1 = i / 4, p2 = p1 + 1, p3 = p1 + 2, p4 = p1 + 3;
      tensor[p1] = (cropped.data[i] - mean0) * inv0;
      tensor[p2] = (cropped.data[i + 4] - mean0) * inv0;
      tensor[p3] = (cropped.data[i + 8] - mean0) * inv0;
      tensor[p4] = (cropped.data[i + 12] - mean0) * inv0;
      tensor[p1 + plane] = (cropped.data[i + 1] - mean1) * inv1;
      tensor[p2 + plane] = (cropped.data[i + 5] - mean1) * inv1;
      tensor[p3 + plane] = (cropped.data[i + 9] - mean1) * inv1;
      tensor[p4 + plane] = (cropped.data[i + 13] - mean1) * inv1;
      tensor[p1 + 2 * plane] = (cropped.data[i + 2] - mean2) * inv2;
      tensor[p2 + 2 * plane] = (cropped.data[i + 6] - mean2) * inv2;
      tensor[p3 + 2 * plane] = (cropped.data[i + 10] - mean2) * inv2;
      tensor[p4 + 2 * plane] = (cropped.data[i + 14] - mean2) * inv2;
    }
    return { tensor, center, scale };
  }

  dispose(): void {
    if (this.session) { this.session.release(); this.session = null; }
  }
}

// ---- InstantHMR pose: square crop → NCHW → 2-input ONNX → 5 outputs ---------

interface InstHmrInference {
  keypoints3d: InstantHMRKeypoint3D[];
  keypoints2d: Array<{ x: number; y: number; id: number; name: string }>;
  mhr: Float32Array;
  shape: Float32Array;
  cam: [number, number, number];
  /** Raw `joints_2d` from the model — 70 joints × 2, normalised to the
   * crop's `[-1, 1]` square. Preserved for `debugJointOffsets`. */
  joints2dNorm: number[][];
  /** Crop box used to feed the model. */
  crop: { x0: number; y0: number; size: number };
  preprocessMs: number;
  inferMs: number;
  postprocessMs: number;
}

class InstantHmrPose {
  private model: InstantHMRModel | null = null;
  private cropCanvas: HTMLCanvasElement | null = null;
  private cropCtx: CanvasRenderingContext2D | null = null;
  private cropImageData: ImageData | null = null;
  private imageBuffer: Float32Array | null = null; // 3 * INPUT_SIZE^2
  private cliffBuffer: Float32Array | null = null; // 3

  constructor(
    private readonly modelUrl: string,
    private readonly bboxExpansion: number,
    private readonly backend: 'wasm' | 'webgl' | 'webgpu' | 'webnn' = 'wasm',
  ) {}

  async init(): Promise<void> {
    this.model = new InstantHMRModel(
      this.modelUrl,
      [INSTANTHMR_INPUT_SIZE, INSTANTHMR_INPUT_SIZE],
      this.backend,
    );
    await this.model.init();

    this.cropCanvas = document.createElement('canvas');
    this.cropCanvas.width = INSTANTHMR_INPUT_SIZE;
    this.cropCanvas.height = INSTANTHMR_INPUT_SIZE;
    this.cropCtx = this.cropCanvas.getContext('2d', { willReadFrequently: true });
    if (!this.cropCtx) throw new Error('2D canvas context unavailable');
    this.cropCtx.imageSmoothingEnabled = true;
    this.cropCtx.imageSmoothingQuality = 'high';
    this.cropImageData = this.cropCtx.createImageData(INSTANTHMR_INPUT_SIZE, INSTANTHMR_INPUT_SIZE);
    this.imageBuffer = new Float32Array(3 * INSTANTHMR_INPUT_SIZE * INSTANTHMR_INPUT_SIZE);
    this.cliffBuffer = new Float32Array(3);
  }

  async infer(canvas: HTMLCanvasElement, box: BBoxRect): Promise<InstHmrInference> {
    if (!this.model || !this.cropCtx || !this.cropImageData || !this.imageBuffer || !this.cliffBuffer) {
      throw new Error('InstantHmrPose not initialised');
    }
    const iw = canvas.width;
    const ih = canvas.height;

    const instBBox: InstHmrBBox = {
      x: box.x1, y: box.y1, w: box.x2 - box.x1, h: box.y2 - box.y1,
    };
    const crop = cropBoxFor(instBBox);
    this.renderCrop(canvas, crop, iw, ih);
    pixelsToNCHW(this.cropImageData.data, this.imageBuffer, 0);
    this.cliffBuffer = cliffCondFor(instBBox, iw, ih);

    const tPre0 = performance.now();
    const out = await this.model.call(this.imageBuffer, this.cliffBuffer);
    const tInf1 = performance.now();

    const tPost0 = performance.now();
    const joints2dPx = denormalizeJoints2D(out.joints2dNorm, crop);
    const cam = out.cam;
    const keypoints3d: InstantHMRKeypoint3D[] = new Array(INSTANTHMR_NUM_JOINTS);
    const keypoints2d: Array<{ x: number; y: number; id: number; name: string }> = new Array(INSTANTHMR_NUM_JOINTS);
    for (let k = 0; k < INSTANTHMR_NUM_JOINTS; k++) {
      keypoints3d[k] = {
        x: out.joints3dLocal[k * 3 + 0] + cam[0],
        y: out.joints3dLocal[k * 3 + 1] + cam[1],
        z: out.joints3dLocal[k * 3 + 2] + cam[2],
        id: k,
        name: INSTANTHMR_JOINT_NAMES[k] as string,
      };
      keypoints2d[k] = {
        x: joints2dPx[k * 2],
        y: joints2dPx[k * 2 + 1],
        id: k,
        name: INSTANTHMR_JOINT_NAMES[k] as string,
      };
    }
    const tPost1 = performance.now();

    // Reshape Float32Array(140) → number[70][2] for debugJointOffsets.
    const joints2dNorm: number[][] = new Array(INSTANTHMR_NUM_JOINTS);
    for (let k = 0; k < INSTANTHMR_NUM_JOINTS; k++) {
      joints2dNorm[k] = [out.joints2dNorm[k * 2], out.joints2dNorm[k * 2 + 1]];
    }

    return {
      keypoints3d,
      keypoints2d,
      mhr: out.mhr,
      shape: out.shape,
      cam: [cam[0], cam[1], cam[2]],
      // Raw model outputs preserved for `debugJointOffsets`. `joints2dNorm`
      // is the per-person crop-space tensor in [-1, 1]; `crop` is the box
      // that turns it back into source-frame pixels via denormalizeJoints2D.
      joints2dNorm,
      crop: { x0: crop.x0, y0: crop.y0, size: crop.size },
      preprocessMs: 0, // counted as the rendering+normalize block above (covered by call() latency)
      inferMs: tInf1 - tPre0,
      postprocessMs: tPost1 - tPost0,
    };
  }

  private renderCrop(src: HTMLCanvasElement, crop: CropBox, iw: number, ih: number): void {
    const ctx = this.cropCtx!;
    const { x0, y0, size } = crop;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, INSTANTHMR_INPUT_SIZE, INSTANTHMR_INPUT_SIZE);
    const sx = Math.max(0, x0);
    const sy = Math.max(0, y0);
    const sx2 = Math.min(iw, x0 + size);
    const sy2 = Math.min(ih, y0 + size);
    const sw = sx2 - sx;
    const sh = sy2 - sy;
    if (sw > 0 && sh > 0) {
      const k = INSTANTHMR_INPUT_SIZE / size;
      ctx.drawImage(
        src as unknown as CanvasImageSource,
        sx, sy, sw, sh,
        (sx - x0) * k, (sy - y0) * k,
        sw * k, sh * k,
      );
    }
    this.cropImageData = ctx.getImageData(0, 0, INSTANTHMR_INPUT_SIZE, INSTANTHMR_INPUT_SIZE);
  }
}

// =============================================================================
// Shared private helpers
// =============================================================================

/** Expand a bbox by `factor × (w, h)` on each side, clamped to image bounds.
 * Used by the detector-stride skip-frame path to keep the reused bboxes
 * large enough to contain the person after one or two video frames of
 * motion. Returns a new bbox; the input is untouched. */
function expandBbox(b: BBoxRect, factor: number, iw: number, ih: number): BBoxRect {
  const w = b.x2 - b.x1;
  const h = b.y2 - b.y1;
  const ex = w * factor;
  const ey = h * factor;
  return {
    x1: Math.max(0, b.x1 - ex),
    y1: Math.max(0, b.y1 - ey),
    x2: Math.min(iw, b.x2 + ex),
    y2: Math.min(ih, b.y2 + ey),
    confidence: b.confidence,
  };
}

function applyNMS(dets: BBoxRect[], iouT: number): BBoxRect[] {
  if (dets.length === 0) return [];
  const sorted = dets.slice().sort((a, b) => b.confidence - a.confidence);
  const selected: BBoxRect[] = [];
  // `Uint8Array` is faster than `Set<number>` for the hot inner loop —
  // V8 can't inline `Set.has()` for monomorphic integer keys; an indexed
  // `used[j] !== 0` access compiles to a single load + compare. For a
  // YOLO run with 8400 anchors this is ~2 ms/frame on WebGL.
  const used = new Uint8Array(sorted.length);
  for (let i = 0; i < sorted.length; i++) {
    if (used[i] !== 0) continue;
    selected.push(sorted[i]);
    used[i] = 1;
    for (let j = i + 1; j < sorted.length; j++) {
      if (used[j] !== 0) continue;
      if (iou(sorted[i], sorted[j]) > iouT) used[j] = 1;
    }
  }
  return selected;
}

function iou(a: BBoxRect, b: BBoxRect): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  if (x2 <= x1 || y2 <= y1) return 0;
  const inter = (x2 - x1) * (y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

function decodeSimCC(
  sx: Float32Array, sy: Float32Array, sz: Float32Array,
  shx: number[], shy: number[], shz: number[],
  center: [number, number], scale: [number, number],
  iw: number, ih: number,
  poseConfidence: number,
  zRange: number,
): Omit<Pose3DInference, 'preprocessMs' | 'inferMs' | 'postprocessMs'> {
  const halfScale0 = scale[0] * 0.5;
  const halfScale1 = scale[1] * 0.5;
  const zHalfRange = zRange * 0.5;

  const numKeypoints = shx[1];
  const wx = shx[2];
  const wy = shy[2];
  const wz = shz[2];
  const invWx = 1 / wx;
  const invWy = 1 / wy;
  const invWz = 1 / wz;

  // Pre-allocate every leaf array once instead of `new Array(numKeypoints)`
  // + per-keypoint `[x, y, z]` literals. For 133 keypoints × 4 nested
  // arrays that removes ~532 small-array allocations per person.
  const keypoints: number[][] = new Array(numKeypoints);
  const scores: number[] = new Array(numKeypoints);
  const keypoints2d: number[][] = new Array(numKeypoints);
  const keypointsSimcc: number[][] = new Array(numKeypoints);
  for (let k = 0; k < numKeypoints; k++) {
    keypoints[k] = new Array<number>(3);
    keypoints2d[k] = new Array<number>(2);
    keypointsSimcc[k] = new Array<number>(3);
  }

  for (let k = 0; k < numKeypoints; k++) {
    // SimCC X argmax (1×wx loop).
    let maxX = sx[k * wx];
    let argmaxX = 0;
    for (let i = 1; i < wx; i++) {
      const v = sx[k * wx + i];
      if (v > maxX) { maxX = v; argmaxX = i; }
    }
    // SimCC Y argmax (1×wy loop).
    let maxY = sy[k * wy];
    let argmaxY = 0;
    for (let i = 1; i < wy; i++) {
      const v = sy[k * wy + i];
      if (v > maxY) { maxY = v; argmaxY = i; }
    }
    // SimCC Z argmax (1×wz loop).
    let maxZ = sz[k * wz];
    let argmaxZ = 0;
    for (let i = 1; i < wz; i++) {
      const v = sz[k * wz + i];
      if (v > maxZ) { maxZ = v; argmaxZ = i; }
    }

    const score = 0.5 * (maxX + maxY);
    scores[k] = score > poseConfidence ? score : 0;

    const normX = argmaxX * invWx;
    const normY = argmaxY * invWy;
    const normZ = argmaxZ * invWz;
    const x = normX * scale[0] - halfScale0 + center[0];
    const y = normY * scale[1] - halfScale1 + center[1];
    const z = normZ * zRange - zHalfRange;

    // Fill the pre-allocated leaf arrays in place — `[a,b,c] = ...` on
    // the hot path forces a fresh allocation each iteration.
    const kp = keypoints[k];
    kp[0] = x; kp[1] = y; kp[2] = z;

    const k2 = keypoints2d[k];
    let cx = x; if (cx < 0) cx = 0; else if (cx > iw) cx = iw;
    let cy = y; if (cy < 0) cy = 0; else if (cy > ih) cy = ih;
    k2[0] = cx; k2[1] = cy;

    const ks = keypointsSimcc[k];
    ks[0] = normX; ks[1] = normY; ks[2] = normZ;
  }
  return { keypoints, scores, keypoints2d, keypointsSimcc };
}

// =============================================================================
// Pose3DDetector — public class
// =============================================================================

/**
 * Pose3DDetector — unified 3D-pose estimation pipeline.
 *
 * The pipeline is composed from three orthogonal config fields
 * (`objectModel`, `pose3dModel`, `backend`) — see `Pose3DDetectorConfig`.
 * 8 combinations are expressible (4 objectModels × 2 pose3dModels); the most
 * common are:
 *
 *   - `objectModel: 'yolov12n',    pose3dModel: 'rtmw3d'`     (default — COCO17 3D)
 *   - `objectModel: 'mediapipe',   pose3dModel: 'rtmw3d'`     (fastest on large frames)
 *   - `objectModel: 'yolov12n',    pose3dModel: 'instanthmr'` (MHR70 mesh)
 *
 * `detectFrom*()` returns `Pose3DResult` when `pose3dModel === 'rtmw3d'`
 * (the generic TS default) and `InstantHMR3DResult` when
 * `pose3dModel === 'instanthmr'`. Field `lastProfile` is populated when
 * `config.profile === true`.
 */
export class Pose3DDetector<T extends Pose3DDetectorConfigGeneric = { pose3dModel: Pose3DModel }> {
  /** Person detector picked by `config.objectModel`. Default `'yolov12n'`.
   * Updated by `setObjectModel()` when `pose3dModel === 'instanthmr'`. */
  objectModel: Pose3DObjectModel;
  /** 3D pose model picked by `config.pose3dModel`. Default `'rtmw3d'`. */
  readonly pose3dModel: Pose3DModel;
  private initialized = false;

  // Pipeline dispatch flags — derived from `objectModel` × `pose3dModel`
  // once at construction. The hot paths branch on these references (not
  // on `this.objectModel` / `this.pose3dModel` strings) to keep the
  // inference loops branch-light.
  private detKind: 'yolo' | 'mediapipe' = 'yolo';
  private poseKind: 'rtmw3d' | 'instanthmr' = 'rtmw3d';

  private yoloDetector: YoloDetector | null = null;
  private mpDetector: MediaPipePersonDetector | null = null;
  private rtmwPose: Rtmw3DPose | null = null;
  private instHmrPose: InstantHmrPose | null = null;
  private _yoloReady = false;
  private _mpReady = false;

  // Detector stride (pose3dModel === 'instanthmr' only). When > 1, the
  // person detector runs every Nth frame; the skipped frames reuse the
  // last bboxes (slightly expanded). Pose inference runs on every frame.
  private _stride = 1;
  private _frameCounter = 0;
  private _lastInstBoxes: BBoxRect[] = [];

  private backend: 'wasm' | 'webgl' | 'webgpu' | 'webnn';
  private cache: boolean;
  private profile: boolean;
  private _debugJointOffsets: boolean;
  private onInitProgress: (stage: string, detail?: string) => void;
  private mediaPipeScoreThreshold: number;

  // Resolved model URLs — defaults stay on HF / Google Storage so callers
  // don't have to thread URLs through every constructor call. Overrides
  // happen via config.detModel / config.poseModel /
  // config.mediaPipeModelPath.
  private detModelUrl: string;
  private poseModelUrl: string;
  private mpModelUrl: string;

  /** Last profile breakdown — `null` until `config.profile === true` and
   * the detector has run at least one detection. */
  lastProfile: Pose3DProfile | null = null;

  constructor(config: Pose3DDetectorConfig) {
    this.objectModel = config.objectModel;
    this.pose3dModel = config.pose3dModel;
    switch (this.objectModel) {
      case 'mediapipe': this.detKind = 'mediapipe'; break;
      default:          this.detKind = 'yolo'; break;
    }
    this.poseKind = this.pose3dModel;

    this.cache = config.cache ?? SHARED_DEFAULT_CONFIG.cache;
    this.profile = config.profile ?? SHARED_DEFAULT_CONFIG.profile;
    this.onInitProgress = config.onInitProgress ?? SHARED_DEFAULT_CONFIG.onInitProgress;

    // Resolve URLs once at construction.
    // - `objectModel` (enum) → URL via `YOLO_VERSIONS`, unless `detModel`
    //   (raw URL) is given — that wins.
    // - `pose3dModel` → RTMW3D URL or InstantHMR URL by default;
    //   `poseModel` overrides.
    if (!config.detModel) {
      if (this.objectModel === 'mediapipe') {
        this.detModelUrl = config.detModel ?? YOLO_MODEL_DEFAULT;
      } else {
        this.detModelUrl = resolveYoloModelUrl(this.objectModel);
      }
    } else {
      this.detModelUrl = config.detModel;
    }
    this.mpModelUrl = config.mediaPipeModelPath ?? MEDIAPIPE_EFFICIENTDET_URL;
    this.poseModelUrl = config.poseModel
      ?? (this.pose3dModel === 'instanthmr' ? INSTANTHMR_MODEL_URL : POSE_MODEL_RTMW3D_DEFAULT);

    // Pose tunables
    this._stride = Math.max(1, Math.floor(config.detectorStride ?? INSTANTHMR_DEFAULT_CONFIG.detectorStride));
    this._frameCounter = 0;
    this._lastInstBoxes = [];
    this._debugJointOffsets = config.debugJointOffsets === true;
    this.mediaPipeScoreThreshold = config.mediaPipeScoreThreshold ?? MEDIAPIPE_DEFAULT_CONFIG.mediaPipeScoreThreshold;

    // Build only the detectors / pose models we need. The instanthmr pose
    // model is special — both detectors (YOLO + MediaPipe) are built
    // eagerly so `setObjectModel()` can flip the person detector at
    // runtime without rebuilding the Pose3DDetector.
    if (this.pose3dModel === 'instanthmr') {
      this.backend = config.backend;
      this.yoloDetector = new YoloDetector({
        detInputSize: config.detInputSize ?? RTMW3D_DEFAULT_CONFIG.detInputSize,
        detConfidence: config.detConfidence ?? RTMW3D_DEFAULT_CONFIG.detConfidence,
        nmsThreshold: config.nmsThreshold ?? RTMW3D_DEFAULT_CONFIG.nmsThreshold,
        backend: this.backend,
      });
      this.mpDetector = new MediaPipePersonDetector({
        mediaPipeScoreThreshold: this.mediaPipeScoreThreshold,
        mediaPipeMaxResults: config.mediaPipeMaxResults ?? INSTANTHMR_DEFAULT_CONFIG.mediaPipeMaxResults,
        personsOnly: config.personsOnly ?? INSTANTHMR_DEFAULT_CONFIG.personsOnly,
        mpInputMaxSize: config.mpInputMaxSize ?? INSTANTHMR_DEFAULT_CONFIG.mpInputMaxSize,
      });
      this.instHmrPose = new InstantHmrPose(
        this.poseModelUrl,
        config.bboxExpansion ?? INSTANTHMR_DEFAULT_CONFIG.bboxExpansion,
        this.backend,
      );
      return;
    }

    // pose3dModel === 'rtmw3d' — single detector + RTMW3D pose.
    if (this.detKind === 'yolo') {
      this.backend = config.backend;
      this.mediaPipeScoreThreshold = 0;
      this.yoloDetector = new YoloDetector({
        detInputSize: config.detInputSize ?? RTMW3D_DEFAULT_CONFIG.detInputSize,
        detConfidence: config.detConfidence ?? RTMW3D_DEFAULT_CONFIG.detConfidence,
        nmsThreshold: config.nmsThreshold ?? RTMW3D_DEFAULT_CONFIG.nmsThreshold,
        backend: this.backend,
      });
    } else {
      this.backend = config.backend;
      this.mpDetector = new MediaPipePersonDetector({
        mediaPipeScoreThreshold: this.mediaPipeScoreThreshold,
        mediaPipeMaxResults: config.mediaPipeMaxResults ?? MEDIAPIPE_DEFAULT_CONFIG.mediaPipeMaxResults,
        personsOnly: config.personsOnly ?? MEDIAPIPE_DEFAULT_CONFIG.personsOnly,
        mpInputMaxSize: config.mpInputMaxSize ?? MEDIAPIPE_DEFAULT_CONFIG.mpInputMaxSize,
      });
    }
    this.rtmwPose = new Rtmw3DPose({
      poseInputSize: config.poseInputSize ?? RTMW3D_DEFAULT_CONFIG.poseInputSize,
      poseConfidence: config.poseConfidence ?? RTMW3D_DEFAULT_CONFIG.poseConfidence,
      zRange: config.zRange ?? RTMW3D_DEFAULT_CONFIG.zRange,
    });
  }

  async init(): Promise<void> {
    if (this.initialized) return;
    this.onInitProgress(
      'start',
      `init Pose3DDetector [objectModel=${this.objectModel}, pose3dModel=${this.pose3dModel}]`,
    );
    // For pose3dModel === 'instanthmr' we eagerly build BOTH detectors so
    // setObjectModel() can swap between them without a rebuild. Init only
    // the one matching the current `objectModel`; the other is lazy-init'd
    // on first swap.
    const eagerInitOther = this.pose3dModel === 'instanthmr';
    const initMp = eagerInitOther || this.detKind === 'mediapipe';
    const initYolo = eagerInitOther || this.detKind === 'yolo';
    if (initMp && this.mpDetector) {
      this.onInitProgress('mp-init', 'loading MediaPipe WASM + EfficientDet-Lite0');
      await this.mpDetector.init(this.mpModelUrl);
      this._mpReady = true;
      this.onInitProgress('mp-ready', 'MediaPipe detector ready');
    }
    if (initYolo && this.yoloDetector) {
      this.onInitProgress('yolo-load', `loading YOLO detector (${this.detModelUrl})`);
      await this.yoloDetector.init(this.detModelUrl, this.cache);
      this._yoloReady = true;
      this.onInitProgress('yolo-ready', 'YOLO detector ready');
    }
    if (this.poseKind === 'rtmw3d' && this.rtmwPose) {
      this.onInitProgress('pose-load', `loading RTMW3D pose model (${this.poseModelUrl})`);
      await this.rtmwPose.init(this.poseModelUrl, this.backend, this.cache);
      this.onInitProgress('pose-ready', 'RTMW3D ready');
    }
    if (this.poseKind === 'instanthmr' && this.instHmrPose) {
      this.onInitProgress('pose-load', `loading InstantHMR pose model (${this.backend})`);
      await this.instHmrPose.init();
      this.onInitProgress('pose-ready', 'InstantHMR ready');
    }
    this.initialized = true;
    this.onInitProgress(
      'ready',
      `Pose3DDetector [objectModel=${this.objectModel}, pose3dModel=${this.pose3dModel}] ready`,
    );
  }

  /** Hot-swap the person detector at runtime. Only works when `pose3dModel
   * === 'instanthmr'` — that's the case where both YOLO and MediaPipe
   * detectors are eagerly built. Throws for `pose3dModel === 'rtmw3d'`;
   * rebuild the Pose3DDetector with a fresh config instead. */
  async setObjectModel(model: Pose3DObjectModel): Promise<void> {
    if (this.pose3dModel !== 'instanthmr') {
      throw new Error(
        `setObjectModel() only applies when pose3dModel === 'instanthmr'; ` +
        `rebuild Pose3DDetector with a new config to change objectModel when pose3dModel === 'rtmw3d'.`,
      );
    }
    const newDetKind = model === 'mediapipe' ? 'mediapipe' : 'yolo';
    if (this.objectModel === model && this.detKind === newDetKind) return;
    this.objectModel = model;
    this.detKind = newDetKind;
    if (newDetKind === 'yolo') {
      // Update the YOLO URL when switching between YOLO versions (the
      // MediaPipe path doesn't touch detModelUrl).
      this.detModelUrl = resolveYoloModelUrl(model);
    }
    if (newDetKind === 'yolo' && this.yoloDetector && !this._yoloReady) {
      await this.yoloDetector.init(this.detModelUrl, this.cache);
      this._yoloReady = true;
    }
    if (newDetKind === 'mediapipe' && this.mpDetector && !this._mpReady) {
      await this.mpDetector.init(this.mpModelUrl);
      this._mpReady = true;
    }
  }

  /**
   * Run 3D pose detection on a Canvas. Returns `Pose3DResult` when
   * `pose3dModel === 'rtmw3d'` (the generic TS default) and
   * `InstantHMR3DResult` when `pose3dModel === 'instanthmr'`.
   */
  async detectFromCanvas(canvas: HTMLCanvasElement): Promise<Pose3DDetectorResult<T>> {
    if (!this.initialized) await this.init();
    if (this.pose3dModel === 'instanthmr') {
      return (await this.detectInstHmrFromCanvas(canvas)) as Pose3DDetectorResult<T>;
    }
    if (this.detKind === 'mediapipe') {
      return (await this.detectRtmwFromCanvasMP(canvas)) as Pose3DDetectorResult<T>;
    }
    return (await this.detectRtmwFromCanvasYOLO(canvas)) as Pose3DDetectorResult<T>;
  }

  async detectFromVideo(video: HTMLVideoElement, target?: HTMLCanvasElement): Promise<Pose3DDetectorResult<T>> {
    if (video.readyState < 2) throw new Error('Video not ready');
    const c = target ?? document.createElement('canvas');
    // Only resize the canvas when its dimensions actually differ from the
    // video — setting `width`/`height` *reallocates* the backing store and
    // discards GPU/CPU buffers. For continuous video the dimensions stay
    // constant, so this is purely overhead.
    if (c.width !== video.videoWidth || c.height !== video.videoHeight) {
      c.width = video.videoWidth;
      c.height = video.videoHeight;
    }
    const ctx = c.getContext('2d', { willReadFrequently: true, alpha: false });
    if (!ctx) throw new Error('Could not get canvas context');
    ctx.drawImage(video, 0, 0);
    return this.detectFromCanvas(c);
  }

  async detectFromImage(image: HTMLImageElement, target?: HTMLCanvasElement): Promise<Pose3DDetectorResult<T>> {
    if (!image.complete || !image.naturalWidth) throw new Error('Image not loaded');
    const c = target ?? document.createElement('canvas');
    c.width = image.naturalWidth;
    c.height = image.naturalHeight;
    const ctx = c.getContext('2d', { willReadFrequently: true, alpha: false });
    if (!ctx) throw new Error('Could not get canvas context');
    ctx.drawImage(image, 0, 0);
    return this.detectFromCanvas(c);
  }

  async detectFromBitmap(bitmap: ImageBitmap, target?: HTMLCanvasElement): Promise<Pose3DDetectorResult<T>> {
    const c = target ?? document.createElement('canvas');
    c.width = bitmap.width;
    c.height = bitmap.height;
    const ctx = c.getContext('2d', { willReadFrequently: true, alpha: false });
    if (!ctx) throw new Error('Could not get canvas context');
    ctx.drawImage(bitmap, 0, 0);
    return this.detectFromCanvas(c);
  }

  async detectFromFile(file: File, target?: HTMLCanvasElement): Promise<Pose3DDetectorResult<T>> {
    const img = await loadImageFromFile(file);
    return this.detectFromImage(img, target);
  }

  async detectFromBlob(blob: Blob, target?: HTMLCanvasElement): Promise<Pose3DDetectorResult<T>> {
    const bitmap = await loadBitmapFromBlob(blob);
    return this.detectFromBitmap(bitmap, target);
  }

  /** Lower-level entry point — rtmw3d pipelines only.
   *
   * Takes raw RGBA bytes (`Uint8Array` view of an `ImageData.data.buffer`) plus
   * the source frame width/height. Useful for code paths that already have
   * the pixels decoded (e.g. video frames piped through WebCodecs).
   *
   * Throws if `pose3dModel === 'instanthmr'` since the InstantHMR path
   * operates on the source canvas to draw the square crop. */
  async detect(rgba: Uint8Array, iw: number, ih: number): Promise<Pose3DResult> {
    if (this.pose3dModel === 'instanthmr') {
      throw new Error('detect(rgba, w, h) is only available when pose3dModel === \'rtmw3d\'; use detectFromCanvas for instanthmr');
    }
    if (!this.initialized) await this.init();
    if (this.detKind === 'mediapipe') {
      // Convert RGBA → canvas-shaped object via a temp canvas so MP can drawImage it.
      const c = document.createElement('canvas');
      c.width = iw; c.height = ih;
      const ctx = c.getContext('2d')!;
      const img = ctx.createImageData(iw, ih);
      img.data.set(rgba);
      ctx.putImageData(img, 0, 0);
      return this.detectRtmwFromCanvasMP(c);
    }
    return this.detectRtmwRaw(rgba, iw, ih);
  }

  /** Update the MediaPipe person detector score threshold (no-op for YOLO). */
  async setScoreThreshold(t: number): Promise<void> {
    if (this.mpDetector) await this.mpDetector.setScoreThreshold(t);
    this.mediaPipeScoreThreshold = t;
  }

  dispose(): void {
    if (this.yoloDetector) this.yoloDetector.dispose();
    if (this.mpDetector) this.mpDetector.dispose();
    if (this.rtmwPose) this.rtmwPose.dispose();
    this.instHmrPose = null;
    this.initialized = false;
  }

  // ---- Internal detection paths (one per pipeline) ------------------------

  private async detectRtmwFromCanvasYOLO(canvas: HTMLCanvasElement): Promise<Pose3DResult> {
    const t0 = performance.now();
    const yolo = this.yoloDetector!;
    const pose = this.rtmwPose!;

    // Use the canvas-direct path: GPU `drawImage` into the cached
    // 640×640 target, then only the 1.2 MB padded pixels come back
    // through `getImageData`. On 1080p frames this is ~80-200 ms
    // faster than `getImageData` on the full source canvas + an
    // extra `drawImage` round-trip in `detect(rgba)`.
    const tDet0 = performance.now();
    const boxes = await yolo.detectCanvas(canvas);
    const tDet1 = performance.now();

    const allK: number[][][] = [];
    const allS: number[][] = [];
    const allSimcc: number[][][] = [];
    const allK2: number[][][] = [];
    let preprocessMs = 0, inferMs = 0, postprocessMs = 0;

    for (const box of boxes) {
      const r = await pose.infer(canvas, box);
      allK.push(r.keypoints);
      allS.push(r.scores);
      allSimcc.push(r.keypointsSimcc);
      allK2.push(r.keypoints2d);
      preprocessMs += r.preprocessMs;
      inferMs += r.inferMs;
      postprocessMs += r.postprocessMs;
    }

    const total = performance.now() - t0;
    const detMs = tDet1 - tDet0;
    return {
      keypoints: allK,
      scores: allS,
      keypointsSimcc: allSimcc,
      keypoints2d: allK2,
      stats: {
        personCount: allK.length,
        detTime: Math.round(detMs),
        poseTime: Math.round(total - detMs),
        totalTime: Math.round(total),
        mpMs: Math.round(detMs),
        preprocessMs: Math.round(preprocessMs),
        inferMs: Math.round(inferMs),
        postprocessMs: Math.round(postprocessMs),
      },
    };
  }

  private async detectRtmwRaw(rgba: Uint8Array, iw: number, ih: number): Promise<Pose3DResult> {
    const t0 = performance.now();
    const yolo = this.yoloDetector!;
    const pose = this.rtmwPose!;
    const tDet0 = performance.now();
    const boxes = await yolo.detect(rgba, iw, ih);
    const tDet1 = performance.now();

    // Single intermediate canvas for the source pixels — passed to `pose.infer`
    // for every detected bbox.
    const srcCanvas = rgbaCanvas(rgba, iw, ih);

    const allK: number[][][] = [];
    const allS: number[][] = [];
    const allSimcc: number[][][] = [];
    const allK2: number[][][] = [];
    let preprocessMs = 0, inferMs = 0, postprocessMs = 0;

    for (const box of boxes) {
      const r = await pose.infer(srcCanvas, box);
      allK.push(r.keypoints);
      allS.push(r.scores);
      allSimcc.push(r.keypointsSimcc);
      allK2.push(r.keypoints2d);
      preprocessMs += r.preprocessMs;
      inferMs += r.inferMs;
      postprocessMs += r.postprocessMs;
    }

    const total = performance.now() - t0;
    const result: Pose3DResult = {
      keypoints: allK,
      scores: allS,
      keypointsSimcc: allSimcc,
      keypoints2d: allK2,
      stats: {
        personCount: allK.length,
        detTime: Math.round(tDet1 - tDet0),
        poseTime: Math.round(total - (tDet1 - tDet0)),
        totalTime: Math.round(total),
        ...(this.profile ? {
          mpMs: 0,
          preprocessMs: +preprocessMs.toFixed(2),
          inferMs: +inferMs.toFixed(2),
          postprocessMs: +postprocessMs.toFixed(2),
        } : {}),
      },
    };
    if (this.profile) {
      this.lastProfile = {
        mpMs: 0,
        preprocessMs: +preprocessMs.toFixed(2),
        inferMs: +inferMs.toFixed(2),
        postprocessMs: +postprocessMs.toFixed(2),
        otherMs: +(total - (tDet1 - tDet0) - preprocessMs - inferMs - postprocessMs).toFixed(2),
        personCount: allK.length,
        totalMs: +total.toFixed(2),
      };
    }
    return result;
  }

  private async detectRtmwFromCanvasMP(canvas: HTMLCanvasElement): Promise<Pose3DResult> {
    const t0 = performance.now();
    const mp = this.mpDetector!;
    const pose = this.rtmwPose!;
    const { boxes, mpMs } = await mp.detect(canvas);

    const allK: number[][][] = [];
    const allS: number[][] = [];
    const allSimcc: number[][][] = [];
    const allK2: number[][][] = [];
    let preprocessMs = 0, inferMs = 0, postprocessMs = 0;

    for (const box of boxes) {
      const r = await pose.infer(canvas, box);
      allK.push(r.keypoints);
      allS.push(r.scores);
      allSimcc.push(r.keypointsSimcc);
      allK2.push(r.keypoints2d);
      preprocessMs += r.preprocessMs;
      inferMs += r.inferMs;
      postprocessMs += r.postprocessMs;
    }

    const total = performance.now() - t0;
    const result: Pose3DResult = {
      keypoints: allK,
      scores: allS,
      keypointsSimcc: allSimcc,
      keypoints2d: allK2,
      stats: {
        personCount: allK.length,
        detTime: Math.round(mpMs),
        poseTime: Math.round(total - mpMs),
        totalTime: Math.round(total),
        ...(this.profile ? {
          mpMs: +mpMs.toFixed(2),
          preprocessMs: +preprocessMs.toFixed(2),
          inferMs: +inferMs.toFixed(2),
          postprocessMs: +postprocessMs.toFixed(2),
        } : {}),
      },
    };
    if (this.profile) {
      this.lastProfile = {
        mpMs: +mpMs.toFixed(2),
        preprocessMs: +preprocessMs.toFixed(2),
        inferMs: +inferMs.toFixed(2),
        postprocessMs: +postprocessMs.toFixed(2),
        otherMs: +(total - mpMs - preprocessMs - inferMs - postprocessMs).toFixed(2),
        personCount: allK.length,
        totalMs: +total.toFixed(2),
      };
    }
    return result;
  }

  private async detectInstHmrFromCanvas(canvas: HTMLCanvasElement): Promise<InstantHMR3DResult> {
    const t0 = performance.now();
    const inst = this.instHmrPose!;

    // Detector stride: run the person detector every Nth frame, reuse the
    // last bboxes (slightly expanded) on the skipped frames so the pose
    // model still has somewhere to look. The original InstantHMR repo uses
    // exactly this pattern to keep end-to-end FPS usable.
    this._frameCounter++;
    const shouldDetect =
      this._stride <= 1 || this._frameCounter % this._stride === 0;

    // Dispatch: YOLO12 vs MediaPipe for person detection. The pose model
    // (InstantHMR) is identical in both branches; only the bbox source
    // changes.
    let boxes: BBoxRect[];
    let mpMs = 0;
    if (shouldDetect) {
      if (this.detKind === 'yolo' && this.yoloDetector) {
        // Canvas-direct path — saves an 8 MB RGBA round-trip on 1080p
        // video frames. The pose model also reads from the same canvas
        // below, so keeping it as a canvas avoids any pixel copy.
        const tDet0 = performance.now();
        boxes = await this.yoloDetector.detectCanvas(canvas);
        mpMs = performance.now() - tDet0;
      } else if (this.mpDetector) {
        ({ boxes, mpMs } = await this.mpDetector.detect(canvas));
      } else {
        boxes = [];
      }
      this._lastInstBoxes = boxes;
    } else {
      // Skip-frame: reuse last bboxes, expand by 5 % per stride-step to
      // absorb the motion that happened between real detections. The
      // 1.2× square crop in InstantHmrPose.renderCrop is forgiving on its
      // own, so this only needs to cover the few pixels of person drift
      // between detections.
      const iw = canvas.width;
      const ih = canvas.height;
      const expand = 0.05 * (this._stride - 1);
      boxes = this._lastInstBoxes.map((b) => expandBbox(b, expand, iw, ih));
      mpMs = 0;
    }

    const persons: InstantHMRPerson[] = [];
    let preprocessMs = 0, inferMs = 0, postprocessMs = 0;

    for (const box of boxes) {
      const r = await inst.infer(canvas, box);
      const person: InstantHMRPerson = {
        bbox: {
          x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2, confidence: box.confidence,
        },
        keypoints3d: r.keypoints3d,
        keypoints2d: r.keypoints2d,
        mhr: r.mhr,
        shape: r.shape,
        cam: r.cam,
      };
      if (this._debugJointOffsets) {
        person.joints2dNorm = r.joints2dNorm;
        person.crop = r.crop;
      }
      persons.push(person);
      preprocessMs += r.preprocessMs;
      inferMs += r.inferMs;
      postprocessMs += r.postprocessMs;
    }

    const total = performance.now() - t0;
    const result: InstantHMR3DResult = {
      persons,
      stats: {
        personCount: persons.length,
        detTime: Math.round(mpMs),
        poseTime: Math.round(total - mpMs),
        totalTime: Math.round(total),
        ...(this.profile ? {
          mpMs: +mpMs.toFixed(2),
          preprocessMs: +preprocessMs.toFixed(2),
          inferMs: +inferMs.toFixed(2),
          postprocessMs: +postprocessMs.toFixed(2),
        } : {}),
      },
    };
    if (this.profile) {
      this.lastProfile = {
        mpMs: +mpMs.toFixed(2),
        preprocessMs: +preprocessMs.toFixed(2),
        inferMs: +inferMs.toFixed(2),
        postprocessMs: +postprocessMs.toFixed(2),
        otherMs: +(total - mpMs - preprocessMs - inferMs - postprocessMs).toFixed(2),
        personCount: persons.length,
        totalMs: +total.toFixed(2),
      };
    }
    return result;
  }
}

// Helper for the YOLO raw-bytes path: builds a canvas from RGBA without
// leaking a creation per call.
function rgbaCanvas(rgba: Uint8Array, iw: number, ih: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = iw;
  c.height = ih;
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(iw, ih);
  img.data.set(rgba);
  ctx.putImageData(img, 0, 0);
  return c;
}

// (INSTANTHMR_MODEL_URL is re-exported from src/models/instanthmr via src/index.ts)
