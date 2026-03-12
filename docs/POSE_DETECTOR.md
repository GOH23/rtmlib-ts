# PoseDetector API

High-performance unified API for real-time person detection and pose estimation.

## Overview

`PoseDetector` combines YOLO12 object detection with RTMW pose estimation in a single, optimized interface. Designed for speed and ease of use with convenient methods for web elements.

**Models are loaded automatically from HuggingFace if not specified.**

## Installation

```bash
npm install rtmlib-ts
```

## Quick Start

### Default Models (Auto-loaded)

```typescript
import { PoseDetector } from 'rtmlib-ts';

// Initialize with default models from HuggingFace
const detector = new PoseDetector();
await detector.init();

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const people = await detector.detectFromCanvas(canvas);
```

### From Canvas

```typescript
import { PoseDetector } from 'rtmlib-ts';

const detector = new PoseDetector({
  detModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  poseModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/rtmpose/end2end.onnx',
});
await detector.init();

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const people = await detector.detectFromCanvas(canvas);
```

### From Video (Real-time)

```typescript
const video = document.getElementById('video') as HTMLVideoElement;
const people = await detector.detectFromVideo(video);
```

### From Image Element

```typescript
const img = document.getElementById('image') as HTMLImageElement;
const people = await detector.detectFromImage(img);
```

### From File Upload

```typescript
const fileInput = document.getElementById('file') as HTMLInputElement;
fileInput.addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) {
    const people = await detector.detectFromFile(file);
  }
});
```

### From Camera (Blob)

```typescript
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
const video = document.querySelector('video');
video.srcObject = stream;

video.addEventListener('play', async () => {
  const people = await detector.detectFromVideo(video);
});
```

## API Reference

### Constructor

```typescript
new PoseDetector(config?: PoseDetectorConfig)
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `detModel` | `string` | optional | Path to YOLO12 detection model |
| `poseModel` | `string` | optional | Path to RTMW pose model |
| `detInputSize` | `[number, number]` | `[416, 416]` | Detection input size |
| `poseInputSize` | `[number, number]` | `[384, 288]` | Pose input size |
| `detConfidence` | `number` | `0.5` | Detection confidence threshold |
| `nmsThreshold` | `number` | `0.45` | NMS IoU threshold |
| `poseConfidence` | `number` | `0.3` | Keypoint visibility threshold |
| `backend` | `'wasm' \| 'webgpu'` | `'wasm'` | Execution backend |
| `cache` | `boolean` | `true` | Enable model caching |

### Default Models

If `detModel` and `poseModel` are not specified, the following default models are used:

- **Detector**: `https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx`
- **Pose**: `https://huggingface.co/demon2233/rtmlib-ts/resolve/main/rtmpose/end2end.onnx`

### Methods

#### `init()`

Initialize both detection and pose models.

```typescript
await detector.init();
```

#### `detectFromCanvas()`

Detect poses from HTMLCanvasElement.

```typescript
async detectFromCanvas(canvas: HTMLCanvasElement): Promise<Person[]>
```

#### `detectFromVideo()`

Detect poses from HTMLVideoElement (for real-time video processing).

```typescript
async detectFromVideo(
  video: HTMLVideoElement,
  targetCanvas?: HTMLCanvasElement
): Promise<Person[]>
```

#### `detectFromImage()`

Detect poses from HTMLImageElement.

```typescript
async detectFromImage(
  image: HTMLImageElement,
  targetCanvas?: HTMLCanvasElement
): Promise<Person[]>
```

#### `detectFromFile()`

Detect poses from File object (for file uploads).

```typescript
async detectFromFile(
  file: File,
  targetCanvas?: HTMLCanvasElement
): Promise<Person[]>
```

#### `detectFromBlob()`

Detect poses from Blob (for camera capture or downloads).

```typescript
async detectFromBlob(
  blob: Blob,
  targetCanvas?: HTMLCanvasElement
): Promise<Person[]>
```

#### `detect()`

Low-level method for raw image data.

```typescript
async detect(
  imageData: Uint8Array,
  width: number,
  height: number
): Promise<Person[]>
```

#### `dispose()`

Release resources and models.

```typescript
detector.dispose();
```

### Types

#### `Person`

```typescript
interface Person {
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
  };
  keypoints: Keypoint[];
  scores: number[];
}
```

#### `Keypoint`

```typescript
interface Keypoint {
  x: number;
  y: number;
  score: number;
  visible: boolean;
  name: string;
}
```

**Keypoint Names (COCO17):**
0. `nose`
1. `left_eye`
2. `right_eye`
3. `left_ear`
4. `right_ear`
5. `left_shoulder`
6. `right_shoulder`
7. `left_elbow`
8. `right_elbow`
9. `left_wrist`
10. `right_wrist`
11. `left_hip`
12. `right_hip`
13. `left_knee`
14. `right_knee`
15. `left_ankle`
16. `right_ankle`

#### `PoseStats`

Performance statistics attached to results:

```typescript
interface PoseStats {
  personCount: number;
  detTime: number;      // Detection time (ms)
  poseTime: number;     // Pose estimation time (ms)
  totalTime: number;    // Total processing time (ms)
}
```

Access via: `(people as any).stats`

## Performance Optimization

### 1. Use WebGPU Backend (if available)

```typescript
const detector = new PoseDetector({
  detModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  poseModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/rtmpose/end2end.onnx',
  backend: 'webgpu',  // Faster than WASM
});
```

### 2. Adjust Input Sizes

Smaller input sizes = faster inference:

```typescript
// Fast (lower accuracy)
const detector = new PoseDetector({
  detInputSize: [416, 416],
  poseInputSize: [256, 192],
});

// Balanced
const detector = new PoseDetector({
  detInputSize: [640, 640],
  poseInputSize: [384, 288],
});
```

### 3. Tune Confidence Thresholds

Higher thresholds = fewer detections but faster:

```typescript
const detector = new PoseDetector({
  detConfidence: 0.6,    // Skip low-confidence detections
  poseConfidence: 0.4,   // Only show confident keypoints
});
```

### 4. Reuse Detector Instance

```typescript
// ❌ Don't create new detector for each frame
const detector = new PoseDetector(config);

// ✅ Reuse same instance
for (const frame of videoFrames) {
  const people = await detector.detect(frame.data, frame.width, frame.height);
}
```

### 5. Batch Processing (for multiple images)

```typescript
// Process images sequentially with same detector
const detector = new PoseDetector(config);
await detector.init();

const results = await Promise.all(
  images.map(img => detector.detect(img.data, img.width, img.height))
);
```

## Complete Example

```typescript
import { PoseDetector } from 'rtmlib-ts';

async function main() {
  // Initialize
  const detector = new PoseDetector({
    detModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
    poseModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/rtmpose/end2end.onnx',
    detInputSize: [640, 640],
    poseInputSize: [384, 288],
    detConfidence: 0.5,
    nmsThreshold: 0.45,
    poseConfidence: 0.3,
    backend: 'wasm',
  });

  await detector.init();

  // Load image
  const response = await fetch('image.jpg');
  const blob = await response.blob();
  const imageBitmap = await createImageBitmap(blob);
  
  const canvas = document.createElement('canvas');
  canvas.width = imageBitmap.width;
  canvas.height = imageBitmap.height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(imageBitmap, 0, 0);
  
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = new Uint8Array(imageData.data);

  // Detect
  const startTime = performance.now();
  const people = await detector.detect(data, canvas.width, canvas.height);
  const endTime = performance.now();

  // Print stats
  const stats = (people as any).stats;
  console.log(`Detected ${stats.personCount} people in ${stats.totalTime}ms`);
  console.log(`  Detection: ${stats.detTime}ms`);
  console.log(`  Pose: ${stats.poseTime}ms`);

  // Draw results
  people.forEach((person, i) => {
    // Draw bounding box
    ctx.strokeStyle = `hsl(${i * 60}, 80%, 50%)`;
    ctx.lineWidth = 2;
    ctx.strokeRect(
      person.bbox.x1,
      person.bbox.y1,
      person.bbox.x2 - person.bbox.x1,
      person.bbox.y2 - person.bbox.y1
    );

    // Draw keypoints
    person.keypoints.forEach(kp => {
      if (!kp.visible) return;
      
      ctx.fillStyle = '#00ff00';
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  // Cleanup
  detector.dispose();
}

main();
```

## Browser Support

| Browser | Version | Backend |
|---------|---------|---------|
| Chrome | 94+ | WASM, WebGPU |
| Edge | 94+ | WASM, WebGPU |
| Firefox | 95+ | WASM |
| Safari | 16.4+ | WASM |

## Performance Benchmarks

Typical inference times on M1 MacBook Pro:

| Configuration | Detection | Pose (per person) | Total (3 people) |
|--------------|-----------|-------------------|------------------|
| WASM, 640×640 | 80ms | 25ms | 155ms |
| WASM, 416×416 | 40ms | 15ms | 85ms |
| WebGPU, 640×640 | 30ms | 10ms | 60ms |

## Troubleshooting

### "Model loading failed"

- Ensure models are accessible via HTTP (not `file://` protocol)
- Use a local server: `python -m http.server 8080`
- Check CORS headers

### "Slow inference"

- Switch to WebGPU backend if available
- Reduce input sizes
- Increase confidence thresholds
- Process every Nth frame instead of all frames

### "No detections"

- Lower `detConfidence` threshold
- Ensure person is visible and reasonably sized
- Check image format (RGB, not grayscale)

## License

Apache 2.0
