# Pose3DDetector API

High-performance 3D pose estimation with YOLOX detector and RTMW3D pose model.

## Overview

`Pose3DDetector` combines YOLOX object detection with RTMW3D 3D pose estimation for full-body 3D keypoint detection. This class provides 3D coordinates (x, y, z) for each keypoint instead of just 2D.

## 🆕 NEW: MediaPipe + RTMW3D (FASTEST!)

For the **fastest 3D pose estimation**, use `MediaPipeObject3DPoseDetector` which combines MediaPipe EfficientDet with RTMW3D:

```typescript
import { MediaPipeObject3DPoseDetector } from 'rtmlib-ts';

// 2-3x faster than YOLO + RTMW3D!
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

**Benefits of MediaPipe + RTMW3D:**
- ⚡ **2-3x faster** than YOLO + 3D Pose
- 🎯 **Accurate 3D pose** from RTMW3D
- 🧠 **Fast detection** from MediaPipe EfficientDet
- 💾 **Lower CPU/GPU load**

See [MediaPipe Detector API](MEDIAPIPE_DETECTOR.md) for more details.

## Installation

```bash
npm install rtmlib-ts
```

## Quick Start

### Basic Usage

```typescript
import { Pose3DDetector } from 'rtmlib-ts';

// Initialize with default models (from HuggingFace)
const detector = new Pose3DDetector();
await detector.init();

// Detect from canvas
const result = await detector.detectFromCanvas(canvas);

// Access 3D keypoints
console.log(`Detected ${result.keypoints.length} people`);
console.log(`3D keypoints: ${result.keypoints[0][0]}`); // [x, y, z] in meters
```

### From Canvas

```typescript
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const result = await detector.detectFromCanvas(canvas);

result.keypoints.forEach((person, i) => {
  person.forEach((kpt, j) => {
    console.log(`Person ${i}, Keypoint ${j}: x=${kpt[0]}, y=${kpt[1]}, z=${kpt[2]}`);
  });
});
```

### From Video (Real-time)

```typescript
const video = document.getElementById('video') as HTMLVideoElement;

video.addEventListener('play', async () => {
  while (!video.paused && !video.ended) {
    const result = await detector.detectFromVideo(video);
    
    // Process 3D keypoints
    result.keypoints.forEach((person, i) => {
      person.forEach((kpt, j) => {
        // kpt = [x, y, z]
        console.log(`Kpt ${j}: [${kpt[0].toFixed(2)}, ${kpt[1].toFixed(2)}, ${kpt[2].toFixed(2)}]`);
      });
    });
    
    await new Promise(resolve => requestAnimationFrame(resolve));
  }
});
```

### From Image Element

```typescript
const img = document.getElementById('image') as HTMLImageElement;
const result = await detector.detectFromImage(img);
```

### From File Upload

```typescript
const fileInput = document.getElementById('file') as HTMLInputElement;
fileInput.addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) {
    const result = await detector.detectFromFile(file);
  }
});
```

### From Camera (Blob)

```typescript
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
const video = document.querySelector('video');
video.srcObject = stream;

video.addEventListener('play', async () => {
  const result = await detector.detectFromVideo(video);
});
```

## API Reference

### Constructor

```typescript
new Pose3DDetector(config?: Pose3DDetectorConfig)
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `detModel` | `string` | optional | Path to YOLOX detection model |
| `poseModel` | `string` | optional | Path to RTMW3D pose model |
| `detInputSize` | `[number, number]` | `[640, 640]` | Detection input size |
| `poseInputSize` | `[number, number]` | `[384, 288]` | Pose input size |
| `detConfidence` | `number` | `0.45` | Detection confidence threshold |
| `nmsThreshold` | `number` | `0.7` | NMS IoU threshold |
| `poseConfidence` | `number` | `0.3` | Keypoint visibility threshold |
| `backend` | `'wasm' \| 'webgpu'` | `'wasm'` | Execution backend |
| `cache` | `boolean` | `true` | Enable model caching |
| `zRange` | `number` | `2.1744869` | Z-axis range in meters |

### Default Models

If `detModel` and `poseModel` are not specified, the following default models are used:

- **Detector**: `https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx`
- **Pose**: `https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx`

### Methods

#### `init()`

Initialize both detection and pose models.

```typescript
await detector.init();
```

#### `detectFromCanvas()`

Detect 3D poses from HTMLCanvasElement.

```typescript
async detectFromCanvas(canvas: HTMLCanvasElement): Promise<Wholebody3DResult>
```

#### `detectFromVideo()`

Detect 3D poses from HTMLVideoElement (for real-time video processing).

```typescript
async detectFromVideo(
  video: HTMLVideoElement,
  targetCanvas?: HTMLCanvasElement
): Promise<Wholebody3DResult>
```

#### `detectFromImage()`

Detect 3D poses from HTMLImageElement.

```typescript
async detectFromImage(
  image: HTMLImageElement,
  targetCanvas?: HTMLCanvasElement
): Promise<Wholebody3DResult>
```

#### `detectFromFile()`

Detect 3D poses from File object (for file uploads).

```typescript
async detectFromFile(
  file: File,
  targetCanvas?: HTMLCanvasElement
): Promise<Wholebody3DResult>
```

#### `detectFromBlob()`

Detect 3D poses from Blob (for camera capture or downloads).

```typescript
async detectFromBlob(
  blob: Blob,
  targetCanvas?: HTMLCanvasElement
): Promise<Wholebody3DResult>
```

#### `detect()`

Low-level method for raw image data.

```typescript
async detect(
  imageData: Uint8Array,
  width: number,
  height: number
): Promise<Wholebody3DResult>
```

#### `dispose()`

Release resources and models.

```typescript
detector.dispose();
```

### Types

#### `Wholebody3DResult`

```typescript
interface Wholebody3DResult {
  keypoints: number[][][];      // [numPeople][numKeypoints][3] - 3D coordinates
  scores: number[][];           // [numPeople][numKeypoints] - confidence scores
  keypointsSimcc: number[][][]; // [numPeople][numKeypoints][3] - normalized SimCC coords
  keypoints2d: number[][][];    // [numPeople][numKeypoints][2] - 2D projection
}
```

#### `Pose3DStats`

Performance statistics attached to results:

```typescript
interface Pose3DStats {
  personCount: number;
  detTime: number;      // Detection time (ms)
  poseTime: number;     // Pose estimation time (ms)
  totalTime: number;    // Total processing time (ms)
}
```

Access via: `(result as any).stats`

### Keypoint Structure

The model outputs 17 COCO keypoints per person:

| Index | Name | 3D Output |
|-------|------|-----------|
| 0 | nose | `[x, y, z]` |
| 1 | left_eye | `[x, y, z]` |
| 2 | right_eye | `[x, y, z]` |
| 3 | left_ear | `[x, y, z]` |
| 4 | right_ear | `[x, y, z]` |
| 5 | left_shoulder | `[x, y, z]` |
| 6 | right_shoulder | `[x, y, z]` |
| 7 | left_elbow | `[x, y, z]` |
| 8 | right_elbow | `[x, y, z]` |
| 9 | left_wrist | `[x, y, z]` |
| 10 | right_wrist | `[x, y, z]` |
| 11 | left_hip | `[x, y, z]` |
| 12 | right_hip | `[x, y, z]` |
| 13 | left_knee | `[x, y, z]` |
| 14 | right_knee | `[x, y, z]` |
| 15 | left_ankle | `[x, y, z]` |
| 16 | right_ankle | `[x, y, z]` |

## Complete Example

```typescript
import { Pose3DDetector } from 'rtmlib-ts';

async function main() {
  // Initialize with default models
  const detector = new Pose3DDetector();
  console.log('Loading models...');
  await detector.init();
  console.log('Models loaded!');

  // Load image
  const img = new Image();
  img.src = 'person.jpg';
  await new Promise(resolve => img.onload = resolve);

  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);

  // Detect 3D poses
  const startTime = performance.now();
  const result = await detector.detectFromCanvas(canvas);
  const endTime = performance.now();

  const stats = (result as any).stats;
  console.log(`Detected ${stats.personCount} people in ${stats.totalTime}ms`);
  console.log(`  Detection: ${stats.detTime}ms`);
  console.log(`  3D Pose: ${stats.poseTime}ms`);

  // Process 3D results
  result.keypoints.forEach((person, personIdx) => {
    console.log(`\nPerson ${personIdx + 1}:`);
    person.forEach((kpt, kptIdx) => {
      const score = result.scores[personIdx][kptIdx];
      if (score > 0.5) {
        console.log(
          `  Keypoint ${kptIdx}: [${kpt[0].toFixed(3)}, ${kpt[1].toFixed(3)}, ${kpt[2].toFixed(3)}] ` +
          `(score: ${score.toFixed(3)})`
        );
      }
    });
  });

  // Draw 2D projection on canvas
  result.keypoints2d.forEach((person, personIdx) => {
    const color = `hsl(${personIdx * 60}, 80%, 50%)`;
    
    person.forEach(([x, y], kptIdx) => {
      const score = result.scores[personIdx][kptIdx];
      if (score > 0.5) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  });

  // Display result
  document.body.appendChild(canvas);
}

main();
```

## Performance Optimization

### 1. Use WebGPU Backend (if available)

```typescript
const detector = new Pose3DDetector({
  backend: 'webgpu',  // Faster than WASM
});
```

### 2. Adjust Input Sizes

Smaller input sizes = faster inference:

```typescript
const detector = new Pose3DDetector({
  detInputSize: [416, 416],  // Faster detection
  poseInputSize: [256, 192], // Faster pose estimation
});
```

### 3. Tune Confidence Thresholds

Higher thresholds = fewer detections but faster:

```typescript
const detector = new Pose3DDetector({
  detConfidence: 0.6,    // Skip low-confidence detections
  poseConfidence: 0.4,   // Only show confident keypoints
});
```

### 4. Reuse Detector Instance

```typescript
// ✅ Reuse same instance for multiple frames
const detector = new Pose3DDetector();
await detector.init();

for (const frame of videoFrames) {
  const result = await detector.detect(frame.data, frame.width, frame.height);
}
```

### 5. Process Every Nth Frame

For real-time video, process every few frames:

```typescript
let frameCount = 0;
const processEvery = 3; // Process every 3rd frame

video.addEventListener('play', async () => {
  while (!video.paused && !video.ended) {
    frameCount++;
    if (frameCount % processEvery === 0) {
      const result = await detector.detectFromVideo(video);
      // Process result
    }
    await new Promise(resolve => requestAnimationFrame(resolve));
  }
});
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

| Configuration | Detection | 3D Pose (per person) | Total (3 people) |
|--------------|-----------|---------------------|------------------|
| WASM, 640×640 + 384×288 | 120ms | 45ms | 255ms |
| WASM, 416×416 + 256×192 | 60ms | 25ms | 135ms |
| WebGPU, 640×640 + 384×288 | 50ms | 20ms | 110ms |

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

### "Z-coordinate seems wrong"

- Z values are in metric scale (meters)
- Range is approximately -1.0 to 1.0 meters from camera
- Z is relative to the person's center

## Custom Models

You can use any compatible ONNX models:

```typescript
const detector = new Pose3DDetector({
  detModel: 'path/to/custom_yolox.onnx',
  poseModel: 'path/to/custom_rtmw3d.onnx',
  detInputSize: [640, 640],
  poseInputSize: [384, 288],
});
```

## License

Apache 2.0
