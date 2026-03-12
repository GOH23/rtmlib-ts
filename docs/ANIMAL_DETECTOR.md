# AnimalDetector API

Animal detection and pose estimation API supporting 30 animal species.

## Overview

`AnimalDetector` combines object detection with pose estimation specifically designed for animals. It supports 30 different animal species and provides both bounding box detection and 17 keypoints pose estimation (COCO format).

## Installation

```bash
npm install rtmlib-ts
```

## Quick Start

### Basic Usage

```typescript
import { AnimalDetector } from 'rtmlib-ts';

// Initialize with default models
const detector = new AnimalDetector();
await detector.init();

// Detect animals
const animals = await detector.detectFromCanvas(canvas);
console.log(`Found ${animals.length} animals`);

animals.forEach(animal => {
  console.log(`${animal.className}: ${animal.bbox.confidence * 100}%`);
});
```

### Detect Specific Animals

```typescript
const detector = new AnimalDetector({
  classes: ['dog', 'cat', 'horse'],  // Only detect these
});
await detector.init();

const animals = await detector.detectFromCanvas(canvas);
```

### From Video

```typescript
const video = document.getElementById('video') as HTMLVideoElement;

video.addEventListener('play', async () => {
  while (!video.paused && !video.ended) {
    const animals = await detector.detectFromVideo(video);
    
    animals.forEach(animal => {
      console.log(`${animal.className} detected`);
    });
    
    await new Promise(resolve => requestAnimationFrame(resolve));
  }
});
```

## Supported Animal Classes

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | gorilla | 15 | polar-bear |
| 1 | spider-monkey | 16 | antelope |
| 2 | howling-monkey | 17 | fox |
| 3 | zebra | 18 | buffalo |
| 4 | elephant | 19 | cow |
| 5 | hippo | 20 | wolf |
| 6 | raccon | 21 | dog |
| 7 | rhino | 22 | sheep |
| 8 | giraffe | 23 | cat |
| 9 | tiger | 24 | horse |
| 10 | deer | 25 | rabbit |
| 11 | lion | 26 | pig |
| 12 | panda | 27 | chimpanzee |
| 13 | cheetah | 28 | monkey |
| 14 | black-bear | 29 | orangutan |

## API Reference

### Constructor

```typescript
new AnimalDetector(config?: AnimalDetectorConfig)
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `detModel` | `string` | optional | Path to detection model |
| `poseModel` | `string` | optional | Path to pose model |
| `detInputSize` | `[number, number]` | `[640, 640]` | Detection input size |
| `poseInputSize` | `[number, number]` | `[256, 192]` | Pose input size |
| `detConfidence` | `number` | `0.5` | Detection confidence threshold |
| `nmsThreshold` | `number` | `0.45` | NMS IoU threshold |
| `poseConfidence` | `number` | `0.3` | Keypoint visibility threshold |
| `backend` | `'wasm' \| 'webgpu'` | `'wasm'` | Execution backend |
| `cache` | `boolean` | `true` | Enable model caching |
| `classes` | `string[] \| null` | `null` | Animal classes to detect |

### Default Models

If models are not specified, the following defaults are used:

- **Detector**: `https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx`
- **Pose**: `https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-b-apt36k.onnx` (ViTPose++-b)

### Available Pose Models

| Model | Input Size | AP (AP10K) | URL |
|-------|------------|------------|-----|
| ViTPose++-s | 256×192 | 74.2 | [vitpose-s-apt36k.onnx](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-s-apt36k.onnx) |
| ViTPose++-b | 256×192 | 75.9 | [vitpose-b-apt36k.onnx](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-b-apt36k.onnx) |
| ViTPose++-l | 256×192 | 80.8 | [vitpose-h-apt36k.onnx](https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-h-apt36k.onnx) |

All ViTPose++ models are trained on 6 datasets and support the 30 animal classes.

### Methods 

#### `init()`

Initialize both detection and pose models.

```typescript
await detector.init();
```

#### `detectFromCanvas()`

Detect animals from HTMLCanvasElement.

```typescript
async detectFromCanvas(canvas: HTMLCanvasElement): Promise<DetectedAnimal[]>
```

#### `detectFromVideo()`

Detect animals from HTMLVideoElement.

```typescript
async detectFromVideo(
  video: HTMLVideoElement,
  targetCanvas?: HTMLCanvasElement
): Promise<DetectedAnimal[]>
```

#### `detectFromImage()`

Detect animals from HTMLImageElement.

```typescript
async detectFromImage(
  image: HTMLImageElement,
  targetCanvas?: HTMLCanvasElement
): Promise<DetectedAnimal[]>
```

#### `detectFromFile()`

Detect animals from File object.

```typescript
async detectFromFile(
  file: File,
  targetCanvas?: HTMLCanvasElement
): Promise<DetectedAnimal[]>
```

#### `detectFromBlob()`

Detect animals from Blob.

```typescript
async detectFromBlob(
  blob: Blob,
  targetCanvas?: HTMLCanvasElement
): Promise<DetectedAnimal[]>
```

#### `setClasses()`

Set which animal classes to detect.

```typescript
detector.setClasses(['dog', 'cat', 'horse']);
```

#### `getAvailableClasses()`

Get list of all supported animal classes.

```typescript
const classes = detector.getAvailableClasses();
console.log(classes); // ['gorilla', 'spider-monkey', ...]
```

#### `dispose()`

Release resources.

```typescript
detector.dispose();
```

### Types

#### `DetectedAnimal`

```typescript
interface DetectedAnimal {
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
  };
  classId: number;
  className: string;
  keypoints: AnimalKeypoint[];
  scores: number[];
}
```

#### `AnimalKeypoint`

```typescript
interface AnimalKeypoint {
  x: number;
  y: number;
  score: number;
  visible: boolean;
  name: string;
}
```

**Keypoint Names (17 COCO format):**
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

#### `AnimalDetectionStats`

Statistics attached to results:

```typescript
interface AnimalDetectionStats {
  animalCount: number;
  classCounts: Record<string, number>;
  detTime: number;
  poseTime: number;
  totalTime: number;
}
```

Access via: `(animals as any).stats`

## Complete Example

```typescript
import { AnimalDetector } from 'rtmlib-ts';

async function main() {
  // Initialize
  const detector = new AnimalDetector({
    classes: ['dog', 'cat', 'horse', 'zebra', 'elephant'],
    detConfidence: 0.5,
    poseConfidence: 0.3,
  });
  
  console.log('Loading models...');
  await detector.init();
  console.log('Models loaded!');
  console.log('Available classes:', detector.getAvailableClasses());

  // Load image
  const img = new Image();
  img.src = 'animals.jpg';
  await new Promise(resolve => img.onload = resolve);

  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);

  // Detect animals
  const startTime = performance.now();
  const animals = await detector.detectFromCanvas(canvas);
  const endTime = performance.now();

  const stats = (animals as any).stats;
  console.log(`Detected ${stats.animalCount} animals in ${stats.totalTime}ms`);
  console.log(`  Detection: ${stats.detTime}ms`);
  console.log(`  Pose: ${stats.poseTime}ms`);
  console.log(`  By class:`, stats.classCounts);

  // Draw results
  animals.forEach((animal, i) => {
    const color = `hsl(${i * 60}, 80%, 50%)`;
    
    // Draw bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(
      animal.bbox.x1,
      animal.bbox.y1,
      animal.bbox.x2 - animal.bbox.x1,
      animal.bbox.y2 - animal.bbox.y1
    );

    // Draw label
    ctx.fillStyle = color;
    ctx.font = '14px sans-serif';
    ctx.fillText(
      `${animal.className} ${(animal.bbox.confidence * 100).toFixed(0)}%`,
      animal.bbox.x1,
      animal.bbox.y1 - 5
    );

    // Draw keypoints
    animal.keypoints.forEach(kp => {
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

## Performance Optimization

### 1. Use WebGPU Backend

```typescript
const detector = new AnimalDetector({
  backend: 'webgpu',  // Faster than WASM
});
```

### 2. Adjust Input Sizes

```typescript
const detector = new AnimalDetector({
  detInputSize: [416, 416],  // Faster detection
  poseInputSize: [192, 256], // Faster pose
});
```

### 3. Filter Classes

```typescript
const detector = new AnimalDetector({
  classes: ['dog', 'cat'],  // Only detect specific animals
});
```

### 4. Tune Confidence Thresholds

```typescript
const detector = new AnimalDetector({
  detConfidence: 0.6,    // Higher = fewer false positives
  poseConfidence: 0.4,   // Only show confident keypoints
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

| Configuration | Detection | Pose (per animal) | Total (3 animals) |
|--------------|-----------|------------------|-------------------|
| WASM, 640×640 + 256×192 | 100ms | 30ms | 190ms |
| WASM, 416×416 + 192×256 | 50ms | 20ms | 110ms |
| WebGPU, 640×640 + 256×192 | 40ms | 15ms | 85ms |

## Troubleshooting

### "No animals detected"

- Lower `detConfidence` threshold
- Ensure animal is visible and reasonably sized
- Check if animal class is in the supported list

### "Slow inference"

- Switch to WebGPU backend
- Reduce input sizes
- Filter to fewer animal classes
- Process every Nth frame in video

### "Model loading failed"

- Ensure models are accessible via HTTP
- Use a local server: `python -m http.server 8080`
- Check CORS headers

## Custom Models

You can use custom models trained on animal datasets:

```typescript
const detector = new AnimalDetector({
  detModel: 'path/to/yolox_animal.onnx',
  poseModel: 'path/to/vitpose_animal.onnx',
  detInputSize: [640, 640],
  poseInputSize: [256, 192],
});
```

## License

Apache 2.0
