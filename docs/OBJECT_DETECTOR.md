# ObjectDetector API

Universal multi-class object detection API supporting all 80 COCO classes.

## Overview

`ObjectDetector` provides a simple, unified interface for detecting objects in images, videos, and live camera feeds. Supports YOLO12 and other YOLO models.

**Model is loaded automatically from HuggingFace if not specified.**

## Installation

```bash
npm install rtmlib-ts
```

## Quick Start

### Default Model (Auto-loaded)

```typescript
import { ObjectDetector } from 'rtmlib-ts';

// Initialize with default model from HuggingFace
const detector = new ObjectDetector({
  classes: ['person', 'car'],  // Optional: filter classes
});
await detector.init();

const objects = await detector.detectFromCanvas(canvas);
console.log(`Found ${objects.length} objects`);
```

### Detect People (Default)

```typescript
import { ObjectDetector } from 'rtmlib-ts';

const detector = new ObjectDetector({
  model: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',  // Path to model file
});
await detector.init();

const objects = await detector.detectFromCanvas(canvas);
console.log(`Found ${objects.length} objects`);
```

### Detect Multiple Classes

```typescript
const detector = new ObjectDetector({
  model: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  classes: ['person', 'car', 'dog'],  // Only detect these
});
await detector.init();

const objects = await detector.detectFromCanvas(canvas);
// Only person, car, and dog detections
```

### Detect All Classes

```typescript
const detector = new ObjectDetector({
  model: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  classes: null,  // Detect all 80 COCO classes
});
await detector.init();

const objects = await detector.detectFromCanvas(canvas);
objects.forEach(obj => {
  console.log(`${obj.className}: ${(obj.confidence * 100).toFixed(1)}%`);
});
```

### Change Classes Dynamically

```typescript
// Start with person detection
const detector = new ObjectDetector({
  model: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  classes: ['person'],
});
await detector.init();

// Later: detect vehicles only
detector.setClasses(['car', 'bus', 'truck']);

// Detect all classes
detector.setClasses(null);
```

## API Reference

### Constructor

```typescript
new ObjectDetector(config?: ObjectDetectorConfig)
```

**Configuration:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | `string` | optional | Path to YOLO model |
| `inputSize` | `[number, number]` | `[416, 416]` | Input size |
| `confidence` | `number` | `0.5` | Confidence threshold |
| `nmsThreshold` | `number` | `0.45` | NMS IoU threshold |
| `classes` | `string[] \| null` | `['person']` | Classes to detect |
| `backend` | `'wasm' \| 'webgpu'` | `'wasm'` | Execution backend |
| `mode` | `'performance' \| 'balanced' \| 'lightweight'` | `'balanced'` | Performance mode |
| `cache` | `boolean` | `true` | Enable model caching |

### Default Model

If `model` is not specified, the following default model is used:

- **Model**: `https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx`

### Methods

#### Detection Methods

```typescript
// From canvas
async detectFromCanvas(canvas: HTMLCanvasElement): Promise<DetectedObject[]>

// From video (real-time)
async detectFromVideo(video: HTMLVideoElement, targetCanvas?: HTMLCanvasElement): Promise<DetectedObject[]>

// From image
async detectFromImage(image: HTMLImageElement, targetCanvas?: HTMLCanvasElement): Promise<DetectedObject[]>

// From file upload
async detectFromFile(file: File, targetCanvas?: HTMLCanvasElement): Promise<DetectedObject[]>

// From blob/camera
async detectFromBlob(blob: Blob, targetCanvas?: HTMLCanvasElement): Promise<DetectedObject[]>

// From raw data
async detect(imageData: Uint8Array, width: number, height: number): Promise<DetectedObject[]>
```

#### Class Management

```typescript
// Set classes to detect
detector.setClasses(['person', 'car']);  // Specific classes
detector.setClasses(null);               // All classes

// Get available classes
const allClasses = detector.getAvailableClasses();  // 80 COCO classes

// Get current filter
const current = detector.getFilteredClasses();
```

#### Other

```typescript
await detector.init();    // Initialize
detector.dispose();       // Cleanup
detector.getStats();      // Get last detection stats
```

### Types

#### `DetectedObject`

```typescript
interface DetectedObject {
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
  };
  classId: number;       // 0-79 for COCO
  className: string;     // e.g., "person", "car"
  confidence: number;    // 0-1
}
```

#### `DetectionStats`

```typescript
interface DetectionStats {
  totalCount: number;
  classCounts: Record<string, number>;
  inferenceTime: number;
}
```

## COCO Classes

All 80 COCO classes (by ID):

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | person | 40 | cup |
| 1 | bicycle | 41 | fork |
| 2 | car | 42 | knife |
| 3 | motorcycle | 43 | spoon |
| 4 | airplane | 44 | bowl |
| 5 | bus | 45 | banana |
| 6 | train | 46 | apple |
| 7 | truck | 47 | sandwich |
| 8 | boat | 48 | orange |
| 9 | traffic light | 49 | broccoli |
| 10 | fire hydrant | 50 | carrot |
| 11 | stop sign | 51 | hot dog |
| 12 | parking meter | 52 | pizza |
| 13 | bench | 53 | donut |
| 14 | bird | 54 | cake |
| 15 | cat | 55 | chair |
| 16 | dog | 56 | couch |
| 17 | horse | 57 | potted plant |
| 18 | sheep | 58 | bed |
| 19 | cow | 59 | dining table |
| 20 | elephant | 60 | toilet |
| 21 | bear | 61 | tv |
| 22 | zebra | 62 | laptop |
| 23 | giraffe | 63 | mouse |
| 24 | backpack | 64 | remote |
| 25 | umbrella | 65 | keyboard |
| 26 | handbag | 66 | cell phone |
| 27 | tie | 67 | microwave |
| 28 | suitcase | 68 | oven |
| 29 | frisbee | 69 | toaster |
| 30 | skis | 70 | sink |
| 31 | snowboard | 71 | refrigerator |
| 32 | sports ball | 72 | book |
| 33 | kite | 73 | clock |
| 34 | baseball bat | 74 | vase |
| 35 | baseball glove | 75 | scissors |
| 36 | skateboard | 76 | teddy bear |
| 37 | surfboard | 77 | hair drier |
| 38 | tennis racket | 78 | toothbrush |
| 39 | bottle | | |

## Examples

### Real-time Video Detection

```typescript
const video = document.querySelector('video')!;
const detector = new ObjectDetector({
  model: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  classes: ['person', 'car'],
});
await detector.init();

async function detectLoop() {
  const objects = await detector.detectFromVideo(video);

  // Draw detections
  objects.forEach(obj => {
    ctx.strokeStyle = '#00ff00';
    ctx.strokeRect(
      obj.bbox.x1, obj.bbox.y1,
      obj.bbox.x2 - obj.bbox.x1,
      obj.bbox.y2 - obj.bbox.y1
    );
  });
  
  requestAnimationFrame(detectLoop);
}

video.play();
detectLoop();
```

### File Upload

```typescript
const fileInput = document.querySelector('input[type="file"]')!;

fileInput.addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  
  const objects = await detector.detectFromFile(file);
  console.log(`Found ${objects.length} objects`);
});
```

### Camera Capture

```typescript
const stream = await navigator.mediaDevices.getUserMedia({
  video: { width: 1280, height: 720 }
});
const video = document.querySelector('video')!;
video.srcObject = stream;

video.addEventListener('play', async () => {
  const objects = await detector.detectFromVideo(video);
  
  // Get stats
  const stats = (objects as any).stats;
  console.log(`Detected ${stats.totalCount} objects in ${stats.inferenceTime}ms`);
  console.log('By class:', stats.classCounts);
});
```

### Performance Optimization

```typescript
// Use WebGPU for faster inference
const detector = new ObjectDetector({
  model: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  backend: 'webgpu',  // Faster than WASM
});

// Smaller input size for speed
const detector = new ObjectDetector({
  model: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  inputSize: [416, 416],  // Faster, less accurate
});

// Higher confidence threshold
const detector = new ObjectDetector({
  model: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  confidence: 0.7,  // Only high-confidence detections
});

// Reuse detector instance
const detector = new ObjectDetector(config);
await detector.init();

// Process multiple frames
for (const frame of frames) {
  const objects = await detector.detectFromCanvas(frame.canvas);
}
```

## Performance

Typical inference times (M1 MacBook Pro):

| Configuration | Time | Classes |
|--------------|------|---------|
| WASM, 640×640 | 80ms | All 80 |
| WASM, 416×416 | 40ms | All 80 |
| WebGPU, 640×640 | 30ms | All 80 |

## Troubleshooting

### "No detections"

- Lower `confidence` threshold (try 0.3)
- Ensure object is visible and large enough
- Check that class is in COCO dataset

### "Unknown class" warning

Class names must match COCO exactly:
- ✅ `'person'`, `'car'`, `'traffic light'`
- ❌ `'Person'`, `'cars'`, `'traffic_light'`

Use `detector.getAvailableClasses()` to see valid names.

### Slow inference

- Use `backend: 'webgpu'` if available
- Reduce `inputSize` to `[416, 416]`
- Increase `confidence` threshold
- Filter to specific `classes`

## License

Apache 2.0
