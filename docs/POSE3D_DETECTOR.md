# Pose3DDetector API

Unified 3D-pose estimation. One class, ten pipelines composed from three orthogonal **required** selectors (`objectModel`, `pose3dModel`, `backend`).

```ts
import { Pose3DDetector } from 'rtmlib-ts';

// YOLOv12 person detector → RTMW3D 17-keypoint 3D pose
const a = new Pose3DDetector({
  objectModel: 'yolov12n',
  pose3dModel: 'rtmw3d',
  backend: 'webgl',
});
await a.init();
const r = await a.detectFromCanvas(canvas);     // Pose3DResult — COCO17

// MediaPipe EfficientDet → RTMW3D — fastest 3D on large frames
const b = new Pose3DDetector({
  objectModel: 'mediapipe',
  pose3dModel: 'rtmw3d',
  backend: 'wasm',
});
await b.init();
const r2 = await b.detectFromCanvas(canvas);    // Pose3DResult

// YOLOv26n → InstantHMR — 70-keypoint MHR mesh (body + hands + face).
const c = new Pose3DDetector({
  objectModel: 'yolo26n',
  pose3dModel: 'instanthmr',
  backend: 'wasm',
});
await c.init();
const r3 = await c.detectFromCanvas(canvas);    // InstantHMR3DResult
```

All ten combinations run on **ONNX Runtime Web** (wasm / webgl / webgpu / webnn). There is no TFLite / LiteRT code path.

## Choosing a pipeline

The pipeline is composed from three orthogonal **required** config fields. Pick one from each axis — every combination is valid:

| Field | Values | Notes |
|---|---|---|
| `objectModel` *(required)* | `'yolov8n' \| 'yolov12n' \| 'yolo26n' \| 'mediapipe'` | Person detector. The three `yolo*` values differ in which Ultralytics YOLO export runs; `'mediapipe'` uses EfficientDet-Lite0. |
| `pose3dModel` *(required)* | `'rtmw3d' \| 'instanthmr'` | 3D pose model. `'rtmw3d'` returns COCO17 keypoints; `'instanthmr'` returns the MHR70 mesh + camera translation + body shape. |
| `backend` *(required)* | `'wasm' \| 'webgl' \| 'webgpu' \| 'webnn'` | ONNX execution provider. The InstantHMR graph is incompatible with WebGL — pick `'wasm'` / `'webgpu'` / `'webnn'` when `pose3dModel === 'instanthmr'`. |

### The 8 combinations

| `objectModel` | `pose3dModel: 'rtmw3d'` | `pose3dModel: 'instanthmr'` |
|---|---|---|
| `'yolov8n'` | YOLOv8n → RTMW3D | YOLOv8n → InstantHMR |
| `'yolov12n'` *(default)* | YOLOv12n → RTMW3D | YOLOv12n → InstantHMR |
| `'yolo26n'` | YOLOv26n → RTMW3D | YOLOv26n → InstantHMR |
| `'mediapipe'` | EfficientDet → RTMW3D | EfficientDet → InstantHMR |

YOLO is roughly an order of magnitude faster than MediaPipe on most frames (YOLOv12n is ~10 MB / ~10 ms on multithreaded WASM vs MediaPipe's ~60 ms). MediaPipe is the right choice on very large frames where `mpInputMaxSize` downscale caps its cost.

### Recommended configurations (Chrome WebGPU benchmark)

Numbers below are **total median ms** for the photo fixture (`examples/photo_detect_pose_3d.png`, 335×719, 1 person) on a real Chrome browser with a discrete GPU and `crossOriginIsolated: true` (COOP+COEP, wasm threads live). Re-run with `npm run bench:chrome-webgpu`. Source JSON: `bench-results/latest.json`.

| Pipeline | backend | total (ms) | speedup vs wasm |
|---|---|---:|---:|
| **yolov8n + rtmw3d**       | webgpu | **27.9**  | **3.87×** |
| yolov8n + rtmw3d           | wasm   | 108.0     | — |
| **yolov8n + instanthmr**   | webgpu | **20.4**  | **4.21×** |
| yolov8n + instanthmr       | wasm   |  86.0     | — |
| **yolo26n + rtmw3d**       | webgpu | **143.3** | **3.30×** |
| yolo26n + rtmw3d           | wasm   | 472.3     | — |
| **yolo26n + instanthmr**   | webgpu | **91.1**  | **2.81×** |
| yolo26n + instanthmr       | wasm   | 256.0     | — |
| yolov12n + rtmw3d          | wasm   | 556.1     | — |
| yolov12n + rtmw3d          | webgpu | 878.4     | **0.63×** ⚠️ |
| yolov12n + instanthmr      | wasm   | 311.0     | — |
| yolov12n + instanthmr      | webgpu | 790.5     | **0.39×** ⚠️ |

**YOLOv8n is the fastest detector on every backend × pose3d model combination.** The surprise winner — it is the smallest detector (~12 MB) and the only one whose pose3D inference time falls under the profile's 1 ms rounding floor, while still being the cheapest YOLO to run on a GPU.

**Recommended combo by use case:**

- **GPU available, want max speed:** `yolov8n + rtmw3d / webgpu` (COCO17 3D, **27.9 ms**) or `yolov8n + instanthmr / webgpu` (MHR70 mesh, **20.4 ms**).
- **GPU available, want larger object coverage / multi-class:** `yolo26n + rtmw3d / webgpu` (143.3 ms) or `yolo26n + instanthmr / webgpu` (91.1 ms).
- **WASM only:** `yolov8n + rtmw3d / wasm` (108 ms — fastest wasm combo). InstantHMR on wasm is `yolo26n` (256 ms).
- **Very large frames:** `mediapipe + rtmw3d / wasm` with `mpInputMaxSize: 640` — the downscale caps MediaPipe detect at ~60 ms regardless of source size.
- **Avoid:** `yolov12n + webgpu` (see Known issues below).

### Known issue: YOLOv12n + WebGPU regression

`objectModel: 'yolov12n'` with `backend: 'webgpu'` is **slower than the same combo on wasm** for every pose3D model in the benchmark:

- `yolov12n + rtmw3d`: wasm 556 ms → webgpu 878 ms (0.63× — **1.6× slower**)
- `yolov12n + instanthmr`: wasm 311 ms → webgpu 790 ms (0.39× — **2.5× slower**)

The regression is entirely in the detector phase (yolov12n det: 138 ms wasm → 765 ms webgpu); the pose3D phase benefits from webgpu as expected. **Root cause:** YOLOv12 is an attention-based architecture (area-attention modules, large-kernel blocks). The ONNX Runtime Web webgpu EP does not have kernel-level optimizations for these ops — many fall back to a generic interpreter that compiles shaders slowly and runs them serially. YOLOv8 and YOLOv26 are CNN-only and have mature webgpu kernels.

If you need webgpu, pick `yolov8n` or `yolo26n`. If you specifically want yolov12n, stay on wasm (or webgl, where yolov12n is the default and works fine).

Per-stage timing breakdown:

```ts
const det = new Pose3DDetector({ objectModel: 'mediapipe', profile: true });
await det.init();
await det.detectFromCanvas(canvas);
console.log(det.lastProfile);
// { mpMs, preprocessMs, inferMs, postprocessMs, otherMs, personCount, totalMs }
```

## Configuration reference

```ts
interface Pose3DDetectorConfig {
  // ---- Pipeline selectors (orthogonal, all required) ----
  objectModel: 'yolov8n' | 'yolov12n' | 'yolo26n' | 'mediapipe';
  pose3dModel: 'rtmw3d' | 'instanthmr';
  backend:     'wasm' | 'webgl' | 'webgpu' | 'webnn';

  // ---- Raw URL overrides (power users / self-hosting) ----
  detModel?:           string;                                       // YOLO URL — wins over objectModel (when it's a yolo*)
  poseModel?:          string;                                       // RTMW3D URL (pose3dModel='rtmw3d') OR InstantHMR URL ('instanthmr')
  mediaPipeModelPath?: string;                                       // EfficientDet-Lite0 URL

  // ---- Detection tunables (apply to whichever detector objectModel picks) ----
  detInputSize?:           [number, number];                         // default [640, 640]
  detConfidence?:          number;                                   // default 0.45
  nmsThreshold?:           number;                                   // default 0.7
  mediaPipeScoreThreshold?: number;                                  // default 0.5 — ignored unless objectModel='mediapipe'
  mediaPipeMaxResults?:    number;                                   // default -1 — ignored unless objectModel='mediapipe'
  mpInputMaxSize?:         number;                                   // default 640 px — ignored unless objectModel='mediapipe'
  personsOnly?:            boolean;                                  // default true — ignored unless objectModel='mediapipe'

  // ---- Pose tunables ----
  poseInputSize?:  [number, number];                                  // RTMW3D: [288, 384] — ignored for instanthmr
  poseConfidence?: number;                                           // RTMW3D: 0.3 — ignored for instanthmr
  zRange?:         number;                                           // RTMW3D: 2.1744869 — ignored for instanthmr
  bboxExpansion?:  number;                                           // instanthmr: 1.2 — only consulted when pose3dModel='instanthmr'

  // ---- instanthmr-only: detector stride ----
  detectorStride?: number;                                           // default 1 — only consulted when pose3dModel='instanthmr'

  // ---- ONNX runtime knobs (all combinations) ----
  deviceType?:      'cpu' | 'gpu' | 'npu';
  powerPreference?: 'default' | 'low-power' | 'high-performance';
  webnnOptions?:    WebNNProviderOptions;

  // ---- Shared ----
  cache?:           boolean;                                         // default true — see "Cache + COEP" below
  profile?:         boolean;                                         // default false → lastProfile
  onInitProgress?:  (stage: string, detail?: string) => void;
}
```

Fields are silently ignored when they don't apply to the chosen pipeline (e.g. `mediaPipeScoreThreshold` is a no-op when `objectModel !== 'mediapipe'`, `bboxExpansion` is only consulted when `pose3dModel === 'instanthmr'`).

### No-URL usage

The library ships sensible defaults — HuggingFace for the ONNX weights and Google Storage for the MediaPipe TFLite. You almost never pass model URLs to the detector constructors:

```ts
// No URLs anywhere — library defaults Just Work.
const det1 = new Pose3DDetector({ objectModel: 'yolov12n', pose3dModel: 'rtmw3d', backend: 'webgl' });
const det2 = new Pose3DDetector({ objectModel: 'yolo26n',   pose3dModel: 'rtmw3d', backend: 'webgl' });
const det3 = new Pose3DDetector({ objectModel: 'yolov12n',  pose3dModel: 'instanthmr', backend: 'wasm' });
const det4 = new Pose3DDetector({ objectModel: 'mediapipe',  pose3dModel: 'rtmw3d', backend: 'wasm' });
```

### Picking a YOLO version

`objectModel: 'yolov8n' | 'yolov12n' | 'yolo26n'` is wired to the central `YOLO_VERSIONS` map exported from the library:

```ts
import { YOLO_VERSIONS, type YoloVersion } from 'rtmlib-ts';

// YOLO_VERSIONS.yolov8n  → /yolo/yolov8n.onnx
// YOLO_VERSIONS.yolov12n → /yolo/yolov12n.onnx  (default; broadest backend coverage)
// YOLO_VERSIONS.yolo26n  → /yolo/yolo26n.onnx  (fastest on modern hardware in headless tests)
```

If both `detModel` (raw URL) and `objectModel` (enum) are given, the raw URL wins. The same `YOLO_VERSIONS` map is shared with `ObjectDetector` and `PoseDetector`, so the same `yoloVersion` enum resolves consistently across the three detectors.

### Detector stride (instanthmr only)

`detectorStride?: number` (default `1`) controls how often the person detector runs. On `n`-th frames the detector runs as usual; on the skipped frames the last detected bboxes are reused (slightly expanded by `0.05 × (stride − 1)`), and the InstantHMR pose model continues to run on every frame. Matches the original InstantHMR repo's `PosePipeline.detector_stride`:

> "The detector is by far the dominant cost on most hardware, so stride 2–3 is the single biggest knob for end-to-end FPS."

Useful when the detector dominates per-frame time (e.g. single-threaded WASM, mobile). On modern hardware with a fast YOLO version (e.g. `yolo26n` on multithreaded WASM) the detector is small enough that stride adds no win — leave it at `1`.

### Hot-swapping the person detector (instanthmr only)

`setObjectModel(model)` switches the instanthmr pipeline's person detector at runtime without rebuilding the `Pose3DDetector`. Both the YOLO detector and the MediaPipe detector are eagerly constructed when `pose3dModel === 'instanthmr'` (the YOLO URL resolves from `model`); the first swap may block on a lazy-init of the not-yet-loaded detector; subsequent swaps are nearly free:

```ts
const det = new Pose3DDetector({
  objectModel: 'yolov12n',
  pose3dModel: 'instanthmr',
  backend:     'wasm',
  profile:     true,
});
await det.init();
await det.detectFromCanvas(canvas);                  // uses yolov12n
await det.setObjectModel('mediapipe');               // hot-swap to MediaPipe
await det.detectFromCanvas(canvas);                  // next frame uses MediaPipe
await det.setObjectModel('yolo26n');                 // hot-swap to YOLOv26n
```

Throws for `pose3dModel === 'rtmw3d'` — that pipeline only constructs the single detector the constructor picked, so the hot-swap isn't possible. Rebuild the `Pose3DDetector` with a new config to change `objectModel` for the rtmw3d pipelines.

Stride cannot be hot-swapped (it's baked into the construction-time state) — change `detectorStride` requires a fresh `Pose3DDetector`.

## Result types

### `Pose3DResult` — `pose3dModel: 'rtmw3d'`

```ts
interface Pose3DResult {
  keypoints:      number[][][];     // [N][K][3]      — 3D in metres (K = model's keypoint count, 133 for cocktail14)
  scores:         number[][];       // [N][K]         — confidence [0, 1]
  keypointsSimcc: number[][][];     // [N][K][3]      — normalised SimCC peaks [0, 1]
  keypoints2d:    number[][][];     // [N][K][2]      — 2D in source pixels
  stats?:         Pose3DStats;      // timing breakdown
}
```

### `InstantHMR3DResult` — `pose3dModel: 'instanthmr'`

```ts
interface InstantHMR3DResult {
  persons: InstantHMRPerson[];
  stats?:  Pose3DStats & { otherMs?: number };
}

interface InstantHMRPerson {
  bbox:        { x1: number; y1: number; x2: number; y2: number; confidence: number };
  keypoints3d: InstantHMRKeypoint3D[];                              // 70 entries, camera frame
  keypoints2d: Array<{ x: number; y: number; id: number; name: string }>;
  mhr:         Float32Array;                                         // length 204 (34 joints × 6D)
  shape:       Float32Array;                                         // length 45 (body + head + hands)
  cam:         [number, number, number];                             // camera translation, metres
}
```

The COCO17 2D skeleton for `Pose3DResult` is exported as `coco17`; the MHR70 layout for InstantHMR is exported as `mhr70` (and `drawMhr70OnCanvas` for quick render).

## API surface

```ts
class Pose3DDetector<T extends { pose3dModel: Pose3DModel } = { pose3dModel: Pose3DModel }> {
  objectModel: Pose3DObjectModel;                    // resolved from config; mutable via setObjectModel
  readonly pose3dModel: Pose3DModel;                 // resolved from config
  lastProfile: Pose3DProfile | null;                 // populated when profile: true

  constructor(config: Pose3DDetectorConfig);

  init(): Promise<void>;
  detectFromCanvas(canvas: HTMLCanvasElement): Promise<Pose3DDetectorResult<T>>;
  detectFromVideo(video: HTMLVideoElement, target?: HTMLCanvasElement): Promise<Pose3DDetectorResult<T>>;
  detectFromImage(image: HTMLImageElement, target?: HTMLCanvasElement): Promise<Pose3DDetectorResult<T>>;
  detectFromBitmap(bitmap: ImageBitmap, target?: HTMLCanvasElement): Promise<Pose3DDetectorResult<T>>;
  detectFromFile(file: File, target?: HTMLCanvasElement): Promise<Pose3DDetectorResult<T>>;
  detectFromBlob(blob: Blob, target?: HTMLCanvasElement): Promise<Pose3DDetectorResult<T>>;

  // Lower-level: pose3dModel='rtmw3d' only. Throws for instanthmr.
  detect(rgba: Uint8Array, width: number, height: number): Promise<Pose3DResult>;

  // Hot-swap person detector for instanthmr (lazy-init). Throws for rtmw3d.
  setObjectModel(model: Pose3DObjectModel): Promise<void>;

  // Updates MediaPipe score threshold (no-op when objectModel !== 'mediapipe').
  setScoreThreshold(threshold: number): Promise<void>;

  dispose(): void;
}
```

TS conditional narrowing on the generic means `detectFromCanvas` is typed correctly for whichever `pose3dModel` was passed to the constructor — `Pose3DDetector<{ pose3dModel: 'instanthmr' }>` returns `InstantHMR3DResult`, the default / `'rtmw3d'` returns `Pose3DResult`. No runtime type checks at call sites.

## Default model URLs

| Model | URL |
|---|---|
| **YOLOv8n**  | `https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov8n.onnx` |
| **YOLOv12n** *(default)* | `https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx` |
| **YOLOv26n** | `https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolo26n.onnx` |
| **RTMW3D-X ONNX** | `https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx` |
| **MediaPipe EfficientDet-Lite0** | `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite` |
| **InstantHMR** | `https://huggingface.co/momolesang/InstantHMR/resolve/main/instanthmr.onnx` (exported as `INSTANTHMR_MODEL_URL`) |

Override any of them via `detModel: '…'` / `poseModel: '…'` / `mediaPipeModelPath: '…'`. The YOLO URLs come from the central `YOLO_VERSIONS` map exported from `rtmlib-ts`. Mirror URLs and self-hosted exports are first-class.

### Cache + COEP

`cache: true` (the default) persists model weights in the Cache API (ONNX) or IndexedDB (MediaPipe `.tflite`). The library uses a cache-first strategy in `getCachedModel()` — every fetch first checks `cache.match(url)`, and only on a miss falls back to the network.

This is what makes the COEP (`Cross-Origin-Embedder-Policy: require-corp`) + HuggingFace combination viable for warm-cache users. The HF CDN doesn't send `Cross-Origin-Resource-Policy`, so a fresh cross-origin fetch under COEP lands with a zero-length body. On a warm cache, the response is served same-origin, COEP passes, and ONNX Runtime Web can use the weights. On a cold cache the first fetch may fail with `EmptyModelResponseError`; the failed response is NOT persisted, so the next reload retries the network fetch — which usually succeeds once the cache is warm.

## Custom ONNX models

```ts
import { YOLO_VERSIONS } from 'rtmlib-ts';

// Self-hosted mirror, MediaPipe + RTMW3D
const det = new Pose3DDetector({
  objectModel:        'mediapipe',
  pose3dModel:        'rtmw3d',
  backend:            'wasm',
  mediaPipeModelPath: 'https://my-mirror.example/efficientdet_lite0.tflite',
  poseModel:          'https://my-mirror.example/rtmw3d-x.onnx',
  cache:              true,
});

// Pick a YOLO version by name (no need to know URLs)
const det2 = new Pose3DDetector({
  objectModel: 'yolo26n',
  pose3dModel: 'rtmw3d',
  backend:     'webgpu',
});
```

## Browser support

| Pipeline | WASM | WebGL | WebGPU | WebNN |
|---|---|---|---|---|
| Any `objectModel` + `pose3dModel: 'rtmw3d'` | ✅ | ✅ (default) | ✅ | ✅ |
| Any `objectModel` + `pose3dModel: 'instanthmr'` | ✅ (default) | ⚠️ graph-incompatible | ✅ (real GPU required) | ✅ |

WASM threads (5–7× speedup) require cross-origin isolation — set `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` on the HTML response.

### Cache write resilience

Some browsers' Cache API implementations refuse to persist very large blobs (RTMW3D-X is ~370 MB; on Chrome 141 the `cache.put()` call throws `QuotaExceededError` or `Unexpected internal error`). `src/core/modelCache.ts` wraps the `cache.put()` in a try/catch — init succeeds with the freshly-fetched bytes, and a `console.warn` line is printed so you know the next page load will re-download. The fix has been in place since the 0.1.0 benchmark pass.

## License

Apache 2.0.
