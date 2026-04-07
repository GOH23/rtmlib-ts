# rtmlib-ts

**Real-time Multi-Person Pose Estimation & Object Detection Library**

TypeScript port of [rtmlib](https://github.com/Tau-J/rtmlib) with YOLO12 and MediaPipe support for browser-based AI inference.

## 🚀 Features

- 🎯 **Object Detection** - 80 COCO classes with YOLO12n or MediaPipe EfficientDet
- 🧘 **Pose Estimation (2D)** - 17 keypoints (COCO) with RTMW or 33 keypoints with MediaPipe BlazePose
- 🎯 **Pose Estimation (3D)** - Full 3D pose with Z-coordinates in meters using RTMW3D-X
- 🐾 **Animal Detection** - 30 animal species with ViTPose++ pose estimation
- 🎮 **MediaPipe Integration** - TFLite backend for faster inference
- ⚡ **Fastest Combo** - MediaPipe + RTMW3D for 2-3x faster 3D pose estimation
- 📹 **Video Support** - Real-time camera & video file processing
- 🌐 **Browser-based** - Pure WebAssembly/WebGL/WebGPU, no backend required
- ⚡ **Fast** - Optimized for ~200ms inference (416×416)
- 🎨 **Beautiful UI** - Modern gradient design in playground

## 📦 Installation

```bash
npm install rtmlib-ts
```

## 🔧 Next.js Integration

This library is designed for **browser-only** environments and requires special handling for Next.js applications.

### ⚠️ Important: Server-Side Rendering (SSR)

rtmlib-ts depends on browser APIs (`window`, `document`, `navigator`) and cannot run during server-side rendering. Use these approaches to integrate with Next.js:

### Method 1: Client Components (Recommended)

```tsx
// app/components/PoseDetector.tsx
'use client';

import { useEffect, useRef, useState } from 'react';
import { PoseDetector as PoseDetectorLib, drawResultsOnCanvas } from 'rtmlib-ts';

export default function PoseDetector() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [detector, setDetector] = useState<PoseDetectorLib | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Initialize detector only in browser
    const poseDetector = new PoseDetectorLib({
    });
    
    poseDetector.init().then(() => {
      setDetector(poseDetector);
      setLoading(false);
    });
  }, []);

  const handleDetect = async () => {
    if (!detector || !canvasRef.current) return;
    
    const results = await detector.detectFromCanvas(canvasRef.current);
    const ctx = canvasRef.current.getContext('2d');
    if (ctx) {
      drawResultsOnCanvas(ctx, results, 'pose');
    }
  };

  if (loading) return <div>Loading detector...</div>;

  return (
    <div>
      <canvas ref={canvasRef} width={640} height={480} />
      <button onClick={handleDetect}>Detect Pose</button>
    </div>
  );
}
```

### Method 2: Dynamic Import with SSR Disabled

```tsx
// app/page.tsx
import dynamic from 'next/dynamic';

const PoseDetector = dynamic(
  () => import('./components/PoseDetector'),
  { ssr: false } // Disable server-side rendering
);

export default function Home() {
  return (
    <main>
      <h1>Pose Detection App</h1>
      <PoseDetector />
    </main>
  );
}
```

### Method 3: Environment Detection Utilities

Use built-in helpers to safely handle SSR:

```tsx
'use client';

import { isBrowser, isSSR, initOnnxRuntimeWeb } from 'rtmlib-ts';

useEffect(() => {
  if (isSSR()) {
    console.log('Running on server - skipping initialization');
    return;
  }

  if (isBrowser()) {
    // Safe to use browser APIs
    initOnnxRuntimeWeb();
    // Initialize detectors here
  }
}, []);
```

### Next.js Configuration (Optional)

If you encounter issues with ONNX Runtime Web WASM files, add to `next.config.js`:

```js
/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config) => {
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };
    return config;
  },
};

module.exports = nextConfig;
```

## 🎮 Quick Start

### 1. Try the Playground

```bash
cd rtmlib-playground-main
npm install
npm run dev

# Open http://localhost:3000
```

### 2. Object Detection (YOLO)

```typescript
import { ObjectDetector, drawResultsOnCanvas } from 'rtmlib-ts';

const detector = new ObjectDetector({
  model: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  classes: ['person', 'car', 'dog'],
  confidence: 0.5,
  inputSize: [416, 416],
  backend: 'webgl',
});
await detector.init();

const results = await detector.detectFromCanvas(canvas);
drawResultsOnCanvas(ctx, results, 'object');
```

### 3. Object Detection (MediaPipe - FASTER!)

```typescript
import { ObjectDetector } from 'rtmlib-ts';

const detector = new ObjectDetector({
  detectorType: 'mediapipe',
  mediaPipeModelPath: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite',
  mediaPipeScoreThreshold: 0.5,
  classes: ['person', 'car'],
});
await detector.init();

const results = await detector.detectFromCanvas(canvas);
```

### 4. Pose Estimation (2D)

```typescript
import { PoseDetector, drawResultsOnCanvas } from 'rtmlib-ts';

const detector = new PoseDetector({
  detModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  poseModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/rtmpose/end2end.onnx',
  detInputSize: [416, 416],
  poseInputSize: [384, 288],
  detConfidence: 0.5,
  poseConfidence: 0.3,
  backend: 'webgl',
});
await detector.init();

const poses = await detector.detectFromCanvas(canvas);
drawResultsOnCanvas(ctx, poses, 'pose');
```

### 5. Pose Estimation (3D) - FASTEST with MediaPipe!

```typescript
import { MediaPipeObject3DPoseDetector } from 'rtmlib-ts';

// MediaPipe + RTMW3D = 2-3x faster than YOLO+3D!
const detector = new MediaPipeObject3DPoseDetector({
  mpScoreThreshold: 0.5,
  poseConfidence: 0.3,
  backend: 'webgpu',
  personsOnly: true,
});
await detector.init();

const result = await detector.detectFromCanvas(canvas);
console.log(result.keypoints[0][0]); // [x, y, z] in meters
```

### 6. Animal Detection

```typescript
import { AnimalDetector } from 'rtmlib-ts';

const detector = new AnimalDetector({
  poseModelType: 'vitpose-b',
  classes: ['dog', 'cat', 'horse'],
  detConfidence: 0.5,
  poseConfidence: 0.3,
  backend: 'webgl',
});
await detector.init();

const animals = await detector.detectFromCanvas(canvas);
```

## 📊 Performance

| Model | Input | Time | Use Case |
|-------|-------|------|----------|
| YOLO12n | 416×416 | ~200ms | Real-time video |
| YOLO12n | 640×640 | ~500ms | High accuracy |
| MediaPipe EfficientDet | 320×320 | ~100ms | Fast detection |
| RTMW Pose | 384×288 | ~100ms | Per person |
| **MediaPipe + RTMW3D** | 320×320 + 384×288 | **~150ms** | **Fastest 3D pose!** |

**Optimization Tips:**
- Use `416×416` for video/real-time
- Use `640×640` for static images
- **MediaPipe + RTMW3D** for fastest 3D pose estimation
- First run is slower (WASM compilation)
- Filter classes to reduce processing
- Use `backend: 'webgpu'` for GPU acceleration

## 🎯 Supported Classes (COCO 80)

**Common:** person, car, dog, cat, bicycle, bus, truck  
**Objects:** bottle, chair, couch, potted plant  
**Animals:** bird, horse, sheep, cow, elephant  
**Full list:** See `COCO_CLASSES` export or use class selector in playground

## 🐾 Animal Detection (30 Species)

**Supported:** dog, cat, horse, zebra, elephant, tiger, lion, panda, cow, sheep, bird, and more!

## 🎨 Drawing Utilities

```typescript
import {
  drawDetectionsOnCanvas,
  drawPoseOnCanvas,
  drawResultsOnCanvas
} from 'rtmlib-ts';

drawResultsOnCanvas(ctx, results, 'object');  // or 'pose', 'pose3d'
```

## 📁 Project Structure

```
rtmlib-ts/
├── src/
│   ├── core/                    # Base utilities
│   │   ├── base.ts              # BaseTool class
│   │   ├── modelCache.ts        # Model caching
│   │   └── preprocessing.ts     # Image preprocessing
│   ├── models/                  # Model implementations
│   │   ├── yolo12.ts            # YOLO12 detector
│   │   ├── rtmpose.ts           # RTMPose model
│   │   └── rtmpose3d.ts         # 3D Pose model
│   ├── solution/                # High-level APIs
│   │   ├── objectDetector.ts    # ObjectDetector (80 COCO)
│   │   ├── poseDetector.ts      # PoseDetector (YOLO + RTMW)
│   │   ├── pose3dDetector.ts    # Pose3DDetector
│   │   ├── animalDetector.ts    # AnimalDetector (ViTPose)
│   │   ├── mediaPipeObjectDetector.ts  # MediaPipe Object Detection
│   │   ├── mediaPipePoseDetector.ts    # MediaPipe Pose Landmarker
│   │   └── mediaPipeObject3DPoseDetector.ts  # MediaPipe + RTMW3D
│   ├── types/                   # TypeScript types
│   └── visualization/           # Canvas drawing
├── docs/                        # API documentation
├── rtmlib-playground-main/      # Next.js demo app
└── README.md
```

## 🧩 Detector Types

### Object Detection
- **YOLO** - YOLO12n ONNX model (accurate)
- **MediaPipe** - EfficientDet TFLite model (fast)

### Pose Estimation
- **YOLO + RTMW** - YOLO12 + RTMWpose (accurate 2D)
- **MediaPipe** - BlazePose with 33 keypoints (fast 2D)
- **YOLO + RTMW3D** - YOLO12 + RTMW3D-X (accurate 3D)
- **MediaPipe + RTMW3D** - EfficientDet + RTMW3D-X (⚡ fastest 3D!)

### Animal Detection
- **ViTPose-S/B/L** - Small/Base/Large models for 30 animal species

## 🐛 Known Issues

- **YOLOv26n**: Requires model re-export (format mismatch)
- **First run**: Slow due to WASM compilation
- **Mobile**: Performance varies by device
- **WebGPU**: Requires browser support (Chrome 113+)

## 📝 License

Apache 2.0

## 🙏 Credits

Based on [rtmlib](https://github.com/Tau-J/rtmlib) by Tao Jiang  
YOLO12 by [Ultralytics](https://ultralytics.com)  
RTMW by [OpenMMLab](https://openmmlab.com)  
MediaPipe by [Google](https://developers.google.com/mediapipe)

## 📚 Documentation

- [ObjectDetector API](docs/OBJECT_DETECTOR.md)
- [PoseDetector API](docs/POSE_DETECTOR.md)
- [Pose3DDetector API](docs/POSE3D_DETECTOR.md)
- [AnimalDetector API](docs/ANIMAL_DETECTOR.md)
- [CustomDetector API](docs/CUSTOM_DETECTOR.md)
