# rtmlib-ts

**Real-time Multi-Person Pose Estimation & Object Detection Library**

TypeScript port of [rtmlib](https://github.com/Tau-J/rtmlib) with YOLO12 and MediaPipe support for browser-based AI inference.

## Features

- **Object Detection** — 80 COCO classes with YOLO12n or MediaPipe EfficientDet
- **Pose Estimation (2D)** — 17 keypoints (COCO) with RTMW or 33 keypoints with MediaPipe BlazePose
- **Pose Estimation (3D)** — One unified `Pose3DDetector` with `type` field: YOLO+RTMW3D, MediaPipe+RTMW3D, or MediaPipe+InstantHMR (70-keypoint MHR mesh)
- **Animal Detection** — 30 animal species with ViTPose++ pose estimation
- **Fastest 3D Combo** — MediaPipe + RTMW3D for the fastest 3D-pose path; InstantHMR when you need per-person mesh recovery
- **Video Support** — Real-time camera & video file processing
- **Browser-based** — Pure WebAssembly / WebGL / WebGPU / WebNN, no backend required
- **Fast** — Optimised for ~200 ms/frame (416×416, wasm threads)

## Installation

```bash
npm install rtmlib-ts
```

## Next.js Integration

This library is **browser-only** (depends on `window`, `document`, `navigator`) and must be isolated from SSR. Three patterns work:

### Client Component (recommended)

```tsx
'use client';

import { useEffect, useRef, useState } from 'react';
import { Pose3DDetector, drawResultsOnCanvas } from 'rtmlib-ts';

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [det, setDet] = useState<Pose3DDetector | null>(null);

  useEffect(() => {
    const d = new Pose3DDetector({
      objectModel: 'mediapipe',
      pose3dModel: 'rtmw3d',
      backend:     'wasm',
      profile:     true,
    });
    d.init().then(() => setDet(d));
  }, []);

  const onClick = async () => {
    if (!det || !canvasRef.current) return;
    const r = await det.detectFromCanvas(canvasRef.current);
    console.log(det.lastProfile, r.keypoints.length);
  };

  return (
    <>
      <canvas ref={canvasRef} width={640} height={480} />
      <button onClick={onClick}>Detect</button>
    </>
  );
}
```

### Dynamic import with SSR off

```tsx
import dynamic from 'next/dynamic';
const Pose3DView = dynamic(() => import('./Pose3DView'), { ssr: false });
```

### SSR check helpers

```ts
import { isBrowser, isSSR, initOnnxRuntimeWeb } from 'rtmlib-ts';

if (!isBrowser()) return;
initOnnxRuntimeWeb();
```

### Optional WASM config (`next.config.js`)

```js
webpack: (config) => {
  config.experiments = { ...config.experiments, asyncWebAssembly: true };
  return config;
}
```

## Quick Start

### 3D pose — pick `objectModel` + `pose3dModel` + `backend`

```ts
import { Pose3DDetector } from 'rtmlib-ts';

// (1) YOLO12 + RTMW3D — COCO17 output
const a = new Pose3DDetector({
  objectModel: 'yolov12n',
  pose3dModel: 'rtmw3d',
  backend: 'webgl',
});

// (2) MediaPipe + RTMW3D — fastest 3D on large frames
const b = new Pose3DDetector({
  objectModel: 'mediapipe',
  pose3dModel: 'rtmw3d',
  backend: 'wasm',
});

// (3) YOLO + InstantHMR — 70-keypoint MHR mesh (body + hands + face)
const c = new Pose3DDetector({
  objectModel: 'yolo26n',
  pose3dModel: 'instanthmr',
  backend: 'wasm',
});

await a.init();
const r = await a.detectFromCanvas(canvas);
console.log(r.keypoints[0][0]);               // [x, y, z] in metres (COCO17)
```

All three pipelines run on ONNX Runtime Web. There is no TFLite code path.

### Object detection

```ts
import { ObjectDetector } from 'rtmlib-ts';

// YOLO12n (default)
const det = new ObjectDetector({
  classes: ['person', 'car'],
  confidence: 0.5,
  backend: 'webgl',
});
await det.init();
const res = await det.detectFromCanvas(canvas);
```

For MediaPipe-backed detection (person-only, fast on large frames):

```ts
const det = new ObjectDetector({
  detectorType: 'mediapipe',
  mediaPipeScoreThreshold: 0.5,
  classes: ['person'],
});
```

### 2D pose

```ts
import { PoseDetector, drawResultsOnCanvas } from 'rtmlib-ts';

const det = new PoseDetector({
  detInputSize: [416, 416],
  poseInputSize: [384, 288],
  detConfidence: 0.5,
  poseConfidence: 0.3,
  backend: 'wasm',
});
await det.init();
const poses = await det.detectFromCanvas(canvas);
drawResultsOnCanvas(ctx, poses, 'pose');
```

### Animal detection (30 species)

```ts
import { AnimalDetector } from 'rtmlib-ts';

const det = new AnimalDetector({ poseModelType: 'vitpose-b' });
await det.init();
const animals = await det.detectFromCanvas(canvas);
```

## Project Structure

```
rtmlib-ts/
├── src/
│   ├── core/                          # Browser-safe primitives
│   │   ├── base.ts                    # BaseTool — ort.InferenceSession wrapper
│   │   ├── environment.ts             # isBrowser / isSSR / createCanvas
│   │   ├── onnxRuntime.ts             # initOnnxRuntimeWeb()
│   │   ├── modelCache.ts              # ONNX model Cache API
│   │   ├── mediaPipeCache.ts          # MediaPipe IndexedDB cache
│   │   ├── preprocessing.ts           # bboxXyxy2cs / topDownAffine / normalizeImage
│   │   └── instanthmrGeometry.ts      # cropBoxFor / cliffCondFor / pixelsToNCHW (MHR70)
│   ├── models/                        # Low-level model wrappers
│   │   ├── yolo12.ts
│   │   ├── yolo26.ts
│   │   └── instanthmr.ts              # 2-input ONNX, 5 outputs
│   ├── solution/                      # User-facing detector classes
│   │   ├── objectDetector.ts          # YOLO or MediaPipe, 80 COCO
│   │   ├── poseDetector.ts            # 2D (RTMW / BlazePose)
│   │   ├── pose3dDetector.ts          # 3D — objectModel × pose3dModel (orthogonal)
│   │   ├── animalDetector.ts          # YOLO + ViTPose++
│   │   ├── mediaPipeObjectDetector.ts
│   │   ├── mediaPipePoseDetector.ts
│   │   └── customDetector.ts
│   ├── types/                         # Shared TypeScript types
│   └── visualization/                 # Canvas drawing helpers
├── docs/                              # Per-detector API docs
├── scripts/                           # Smoke tests + 3D pose benchmark
└── README.md
```

## Detector families

| Family | Class | Notes |
|---|---|---|
| Object detection | `ObjectDetector` | `detectorType: 'yolo' \| 'mediapipe'` |
| 2D pose | `PoseDetector` | YOLO + RTMW or MediaPipe BlazePose |
| **3D pose** | `Pose3DDetector` | `objectModel: 'yolov{8,12,26}n' \| 'mediapipe'` × `pose3dModel: 'rtmw3d' \| 'instanthmr'` — single class, all ONNX |
| Animal detection | `AnimalDetector` | YOLO + ViTPose++, 30 species |
| MediaPipe direct | `MediaPipeObjectDetector`, `MediaPipePoseDetector` | Use when you need MediaPipe's standalone output. |

## Known Issues

- **YOLOv26n** — model export format mismatch; the `YOLO26` class is shipped but flagged for re-export.
- **First inference** is slow due to WASM compilation (cold start, not a bug).
- **WebGPU in headless Chromium** — no GPU adapter even with `--enable-unsafe-webgpu`. To exercise the WebGPU path, launch against `channel: 'chrome'` on a host with a real GPU.
- **InstantHMR** — WebGL backend is incompatible with the graph (use `wasm` or `webgpu`).
- **YOLOv12n + WebGPU is slower than wasm** — YOLOv12 is attention-based and the ONNX Runtime Web webgpu EP lacks optimized kernels for it. Pick `yolov8n` or `yolo26n` for webgpu. See `docs/POSE3D_DETECTOR.md` for the benchmark.
- **RTMW3D cache write** — 370 MB model can exceed some browsers' Cache API quotas; `modelCache.ts` logs a warning and serves the bytes anyway, so init still succeeds (next page reload re-downloads).

## Credits

Based on [rtmlib](https://github.com/Tau-J/rtmlib) by Tao Jiang  
YOLO12 by [Ultralytics](https://ultralytics.com)  
RTMW by [OpenMMLab](https://openmmlab.com)  
InstantHMR / MHR70 by [Meta](https://github.com/facebookresearch/instanthmr)  
MediaPipe by [Google](https://developers.google.com/mediapipe)

## Documentation

- [ObjectDetector API](docs/OBJECT_DETECTOR.md)
- [PoseDetector API](docs/POSE_DETECTOR.md)
- [Pose3DDetector API](docs/POSE3D_DETECTOR.md)
- [AnimalDetector API](docs/ANIMAL_DETECTOR.md)
- [CustomDetector API](docs/CUSTOM_DETECTOR.md)
