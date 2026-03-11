# rtmlib-ts

**Real-time Multi-Person Pose Estimation & Object Detection Library**

TypeScript port of [rtmlib](https://github.com/Tau-J/rtmlib) with YOLO12 support for browser-based AI inference.

## 🚀 Features

- 🎯 **Object Detection** - 80 COCO classes with YOLO12n
- 🧘 **Pose Estimation** - 17 keypoints skeleton tracking
- 📹 **Video Support** - Real-time camera & video file processing
- 🌐 **Browser-based** - Pure WebAssembly, no backend required
- ⚡ **Fast** - Optimized for ~200ms inference (416×416)
- 🎨 **Beautiful UI** - Modern gradient design

## 📦 Installation

```bash
npm install rtmlib-ts
```

## 🎮 Quick Start

### 1. Try the Playground

```bash
cd playground
npm install
npm run dev

# Open http://localhost:3000
```

### 2. Object Detection

```typescript
import { ObjectDetector, drawResultsOnCanvas } from 'rtmlib-ts';

// Initialize
const detector = new ObjectDetector({
  model: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  classes: ['person', 'car', 'dog'],  // Filter classes or null for all
  confidence: 0.5,
  inputSize: [416, 416],  // 416 for speed, 640 for accuracy
  backend: 'wasm',
});

await detector.init();

// Detect from canvas
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const results = await detector.detectFromCanvas(canvas);

// Draw results
const ctx = canvas.getContext('2d')!;
drawResultsOnCanvas(ctx, results, 'object');

console.log(`Found ${results.length} objects`);
results.forEach(obj => {
  console.log(`${obj.className}: ${(obj.confidence * 100).toFixed(1)}%`);
});
```

### 3. Pose Estimation

```typescript
import { PoseDetector, drawResultsOnCanvas } from 'rtmlib-ts';

// Initialize
const poseDetector = new PoseDetector({
  detModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  poseModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/rtmpose/end2end.onnx',
  detInputSize: [416, 416],
  detConfidence: 0.5,
  poseConfidence: 0.3,
  backend: 'wasm',
});

await poseDetector.init();

// Detect poses
const results = await poseDetector.detectFromCanvas(canvas);

// Draw skeleton
const ctx = canvas.getContext('2d')!;
drawResultsOnCanvas(ctx, results, 'pose');

console.log(`Found ${results.length} people`);
results.forEach(person => {
  const visibleKpts = person.keypoints.filter(k => k.visible).length;
  console.log(`Person: ${visibleKpts}/17 keypoints visible`);
});
```

### 4. Real-time Video

```typescript
import { ObjectDetector } from 'rtmlib-ts';

const detector = new ObjectDetector({
  model: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  inputSize: [416, 416],  // Faster for video
});
await detector.init();

// Camera stream
const video = document.querySelector('video')!;
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
video.srcObject = stream;

// Detection loop (every 500ms)
setInterval(async () => {
  const results = await detector.detectFromVideo(video);
  console.log(`Detected: ${results.map(r => r.className).join(', ')}`);
}, 500);
```

### 5. Image Upload

```typescript
// File input
<input type="file" accept="image/*" onChange={handleFile} />

// Handler
const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
  const file = e.target.files?.[0];
  if (!file) return;
  
  const results = await detector.detectFromFile(file);
  console.log(`Found ${results.length} objects`);
};
```

## 📊 Performance

| Model | Input | Time | Use Case |
|-------|-------|------|----------|
| YOLO12n | 416×416 | ~200ms | Real-time video |
| YOLO12n | 640×640 | ~500ms | High accuracy |
| RTMW Pose | 384×288 | ~100ms | Per person |

**Optimization Tips:**
- Use 416×416 for video/real-time
- Use 640×640 for static images
- First run is slower (WASM compilation)
- Filter classes to reduce processing

## 🎯 Supported Classes (COCO 80)

**Common:** person, car, dog, cat, bicycle, bus, truck  
**Objects:** bottle, chair, couch, potted plant  
**Animals:** bird, horse, sheep, cow, elephant  
**Full list:** See `COCO_CLASSES` export

## 🎨 Drawing Utilities

```typescript
import { 
  drawDetectionsOnCanvas,
  drawPoseOnCanvas,
  drawResultsOnCanvas  // Universal
} from 'rtmlib-ts';

// Auto-detects mode
drawResultsOnCanvas(ctx, results, 'object');  // or 'pose'

// Custom drawing
drawDetectionsOnCanvas(ctx, detections, '#00ff00');
drawPoseOnCanvas(ctx, people, 0.3);  // 0.3 confidence threshold
```

## 📁 Project Structure

```
rtmlib-ts/
├── src/
│   ├── solution/
│   │   ├── objectDetector.ts   # Object detection
│   │   └── poseDetector.ts     # Pose estimation
│   └── visualization/
│       └── draw.ts             # Canvas utilities
├── playground/                  # Next.js demo
└── models/
    ├── yolo/yolov12n.onnx      # Detector
    └── rtmpose/end2end.onnx    # Pose model
```

## 🐛 Known Issues

- **YOLOv26n**: Requires model re-export (format mismatch)
- **First run**: Slow due to WASM compilation
- **Mobile**: Performance varies by device

## 📝 License

Apache 2.0

## 🙏 Credits

Based on [rtmlib](https://github.com/Tau-J/rtmlib) by Tao Jiang  
YOLO12 by [Ultralytics](https://ultralytics.com)  
RTMW by [OpenMMLab](https://openmmlab.com)
