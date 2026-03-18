# MediaPipe Detectors API

**MediaPipe Tasks Vision** integration for fast object detection and pose estimation in the browser.

## Overview

rtmlib-ts now supports **MediaPipe Tasks Vision** as an alternative backend for object detection and pose estimation. MediaPipe models (TFLite format) provide faster inference compared to ONNX models, making them ideal for real-time applications.

## 🚀 Quick Start

### Object Detection with MediaPipe

```typescript
import { MediaPipeObjectDetector } from 'rtmlib-ts';

const detector = new MediaPipeObjectDetector({
  modelPath: 'https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0.tflite',
  scoreThreshold: 0.5,
  maxResults: -1,  // -1 for all results
  categoryAllowlist: ['person', 'car'],  // Optional: filter classes
});
await detector.init();

const results = await detector.detectFromCanvas(canvas);
console.log(`Found ${results.length} objects`);
```

### Pose Estimation with MediaPipe

```typescript
import { MediaPipePoseDetector } from 'rtmlib-ts';

const detector = new MediaPipePoseDetector({
  modelPath: 'https://storage.googleapis.com/mediapipe-tasks/pose_landmarker/pose_landmarker_full.task',
  numPoses: 3,
  minPoseDetectionConfidence: 0.5,
  minPosePresenceConfidence: 0.5,
});
await detector.init();

const poses = await detector.detectFromCanvas(canvas);
console.log(`Found ${poses.length} people`);
// 33 landmarks per pose with world coordinates (in meters)
```

### 3D Pose with MediaPipe + RTMW3D (FASTEST!)

```typescript
import { MediaPipeObject3DPoseDetector } from 'rtmlib-ts';

// Best combination: MediaPipe for detection (fast) + RTMW3D for 3D pose (accurate)
const detector = new MediaPipeObject3DPoseDetector({
  mpScoreThreshold: 0.5,
  poseConfidence: 0.3,
  backend: 'webgpu',
  personsOnly: true,  // Detect only people
});
await detector.init();

const result = await detector.detectFromCanvas(canvas);
console.log(result.keypoints[0][0]); // [x, y, z] - 3D coordinates in meters
```

## 📦 Installation

```bash
npm install rtmlib-ts
```

## 🎯 MediaPipeObjectDetector

### Constructor Options

```typescript
new MediaPipeObjectDetector(config?: MediaPipeObjectDetectorConfig)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `modelPath` | `string` | required | Path to TFLite model |
| `scoreThreshold` | `number` | `0.5` | Confidence threshold |
| `maxResults` | `number` | `-1` | Max detections (-1 for all) |
| `categoryAllowlist` | `string[]` | `undefined` | Classes to detect |

### Default Model

```
https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0.tflite
```

### Methods

```typescript
await detector.init();
const results = await detector.detectFromCanvas(canvas);
const results = await detector.detectFromVideo(video);
const results = await detector.detectFromImage(image);
```

### Example: Full Object Detection

```typescript
import { MediaPipeObjectDetector, drawDetectionsOnCanvas } from 'rtmlib-ts';

const detector = new MediaPipeObjectDetector({
  modelPath: 'https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0.tflite',
  scoreThreshold: 0.5,
  maxResults: -1,
  categoryAllowlist: ['person', 'car', 'dog'],
});
await detector.init();

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const results = await detector.detectFromCanvas(canvas);

const ctx = canvas.getContext('2d')!;
drawDetectionsOnCanvas(ctx, results);

results.forEach(obj => {
  console.log(`${obj.className}: ${(obj.confidence * 100).toFixed(1)}%`);
});
```

## 🧘 MediaPipePoseDetector

### Constructor Options

```typescript
new MediaPipePoseDetector(config?: MediaPipePoseDetectorConfig)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `modelPath` | `string` | required | Path to .task model |
| `numPoses` | `number` | `3` | Maximum poses to detect |
| `minPoseDetectionConfidence` | `number` | `0.5` | Detection confidence |
| `minPosePresenceConfidence` | `number` | `0.5` | Presence confidence |
| `minTrackingConfidence` | `number` | `0.5` | Tracking confidence |

### Default Model

```
https://storage.googleapis.com/mediapipe-tasks/pose_landmarker/pose_landmarker_full.task
```

### Keypoints (33 BlazePose)

MediaPipe BlazePose provides 33 keypoints:

0-10: Face (nose, eyes, ears, mouth)  
11-22: Upper body (shoulders, elbows, wrists, hips)  
23-32: Lower body (knees, ankles, heels, foot indices)

### Example: Pose Estimation

```typescript
import { MediaPipePoseDetector, drawPoseOnCanvas } from 'rtmlib-ts';

const detector = new MediaPipePoseDetector({
  modelPath: 'https://storage.googleapis.com/mediapipe-tasks/pose_landmarker/pose_landmarker_full.task',
  numPoses: 3,
  minPoseDetectionConfidence: 0.5,
});
await detector.init();

const poses = await detector.detectFromCanvas(canvas);

const ctx = canvas.getContext('2d')!;
drawPoseOnCanvas(ctx, poses);

// Access world coordinates (in meters)
poses.forEach((pose, i) => {
  console.log(`Person ${i}:`);
  pose.keypoints.forEach((kp, j) => {
    console.log(`  ${kp.name}: x=${kp.x?.toFixed(2)}, y=${kp.y?.toFixed(2)}, z=${kp.z?.toFixed(2)}`);
  });
});
```

## 🎯 MediaPipeObject3DPoseDetector

**FASTEST 3D Pose Estimation** - Combines MediaPipe EfficientDet for object detection with RTMW3D-X for accurate 3D pose estimation.

### Why MediaPipe + RTMW3D?

- ⚡ **2-3x faster** than YOLO + 3D Pose
- 🎯 **Accurate 3D pose** from RTMW3D
- 🧠 **Fast detection** from MediaPipe EfficientDet
- 💾 **Lower CPU/GPU load**

### Constructor Options

```typescript
new MediaPipeObject3DPoseDetector(config?: MediaPipeObject3DPoseDetectorConfig)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mpScoreThreshold` | `number` | `0.5` | MediaPipe detection threshold |
| `poseConfidence` | `number` | `0.3` | RTMW3D pose threshold |
| `backend` | `'wasm' \| 'webgpu'` | `'wasm'` | Execution backend |
| `personsOnly` | `boolean` | `true` | Detect only people |

### Example: Fastest 3D Pose

```typescript
import { MediaPipeObject3DPoseDetector } from 'rtmlib-ts';

const detector = new MediaPipeObject3DPoseDetector({
  mpScoreThreshold: 0.5,
  poseConfidence: 0.3,
  backend: 'webgpu',
  personsOnly: true,
});
await detector.init();

const result = await detector.detectFromCanvas(canvas);

// Access 3D keypoints
result.keypoints.forEach((person, i) => {
  console.log(`Person ${i}:`);
  person.forEach((kpt, j) => {
    console.log(`  Keypoint ${j}: [${kpt[0].toFixed(3)}, ${kpt[1].toFixed(3)}, ${kpt[2].toFixed(3)}]m`);
  });
});

// Access 2D projection
result.keypoints2d.forEach((person, i) => {
  person.forEach(([x, y], j) => {
    console.log(`  2D Keypoint ${j}: [${x.toFixed(1)}, ${y.toFixed(1)}]`);
  });
});
```

### Result Structure

```typescript
interface MediaPipeObject3DPoseResult {
  keypoints: number[][][];      // [numPeople][numKeypoints][3] - 3D coords
  keypoints2d: number[][][];    // [numPeople][numKeypoints][2] - 2D projection
  scores: number[][];           // [numPeople][numKeypoints] - confidence
  bboxes: Array<{             // Detection boxes
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
  }>;
}
```

## 📊 Performance Comparison

| Configuration | Time | Accuracy |
|--------------|------|----------|
| MediaPipe Object Detection | ~100ms | Good |
| MediaPipe Pose (BlazePose) | ~80ms | Good (33 keypoints) |
| **MediaPipe + RTMW3D** | **~150ms** | **Excellent (3D)** |
| YOLO12 + RTMW3D | ~350ms | Excellent (3D) |

## 🌐 Available MediaPipe Models

### Object Detection
- **EfficientDet-Lite0**: `https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0.tflite`
- **EfficientDet-Lite2**: `https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite2.tflite`

### Pose Landmarker
- **Pose Landmarker Full**: `https://storage.googleapis.com/mediapipe-tasks/pose_landmarker/pose_landmarker_full.task`
- **Pose Landmarker Heavy**: `https://storage.googleapis.com/mediapipe-tasks/pose_landmarker/pose_landmarker_heavy.task`
- **Pose Landmarker Lite**: `https://storage.googleapis.com/mediapipe-tasks/pose_landmarker/pose_landmarker_lite.task`

## 🔧 Using with Main API

### Object Detection via ObjectDetector

```typescript
import { ObjectDetector } from 'rtmlib-ts';

const detector = new ObjectDetector({
  detectorType: 'mediapipe',  // or 'yolo'
  mediaPipeModelPath: 'https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0.tflite',
  mediaPipeScoreThreshold: 0.5,
  classes: ['person', 'car'],
});
await detector.init();
```

### Pose Detection via PoseDetector

```typescript
import { PoseDetector } from 'rtmlib-ts';

const detector = new PoseDetector({
  detectorType: 'mediapipe',  // or 'yolo-rtmpose'
  mediaPipeModelPath: 'https://storage.googleapis.com/mediapipe-tasks/pose_landmarker/pose_landmarker_full.task',
  mediaPipeNumPoses: 3,
});
await detector.init();
```

## 🎨 Visualization

```typescript
import { drawResultsOnCanvas } from 'rtmlib-ts';

// Works with all detector types
drawResultsOnCanvas(ctx, results, 'object');  // or 'pose', 'pose3d'
```

## 🐛 Troubleshooting

### "Model loading failed"
- Ensure model URL is accessible (CORS enabled)
- Use HTTPS or localhost
- Check network tab for 404 errors

### "Slow inference"
- Use `backend: 'webgpu'` for GPU acceleration
- Reduce input image size
- Use MediaPipe Lite models

### "No detections"
- Lower `scoreThreshold` (try 0.3)
- Ensure object is visible and large enough
- Check class names match MediaPipe categories

## 📚 Related Documentation

- [ObjectDetector API](OBJECT_DETECTOR.md)
- [PoseDetector API](POSE_DETECTOR.md)
- [Pose3DDetector API](POSE3D_DETECTOR.md)

## 📝 License

Apache 2.0
