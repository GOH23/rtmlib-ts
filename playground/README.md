# rtmlib-ts Playground

Interactive Next.js app for testing rtmlib-ts library features.

## Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Open in browser
http://localhost:3000
```

## Features

### Object Detection
- Detect any of 80 COCO classes
- Filter by specific classes (person, car, dog, etc.)
- Real-time camera support
- Image upload

### Pose Estimation
- 17 keypoints per person
- Skeleton visualization
- Real-time camera support
- Image upload

## Usage

1. **Select Mode**: Choose between Object Detection or Pose Estimation
2. **Choose Input**: 
   - Click "Use Camera" for live detection
   - Click "Upload Image" to process a file
3. **Select Classes** (Object Detection only):
   - Check specific classes to detect
   - Uncheck all to detect all 80 classes
4. **Click Detect**: Run inference and see results

## Models

Models are served from `/public/models/`:

```
public/models/
├── yolo/
│   ├── yolov12n.onnx      # Object detection
│   └── yolov26n.onnx
└── rtmpose/
    └── end2end.onnx       # Pose estimation
```

## Tech Stack

- **Framework**: Next.js 15 (App Router)
- **Library**: rtmlib-ts (local import)
- **Backend**: ONNX Runtime Web (WASM)
- **Styling**: Inline CSS (no dependencies)

## Code Example

```typescript
import { ObjectDetector, PoseDetector } from 'rtmlib-ts';

// Object Detection
const detector = new ObjectDetector({
  model: '/models/yolo/yolov12n.onnx',
  classes: ['person', 'car'],
});
await detector.init();
const objects = await detector.detectFromCanvas(canvas);

// Pose Estimation
const poseDetector = new PoseDetector({
  detModel: '/models/yolo/yolov12n.onnx',
  poseModel: '/models/rtmpose/end2end.onnx',
});
await poseDetector.init();
const people = await poseDetector.detectFromCanvas(canvas);
```

## Performance

Expected inference times (varies by device):

| Mode | Input | Time |
|------|-------|------|
| Object (WASM) | 640×640 | ~80ms |
| Object (WebGPU) | 640×640 | ~30ms |
| Pose (WASM) | 640×640 | ~150ms |
| Pose (WebGPU) | 640×640 | ~60ms |

## Troubleshooting

### "Models not found"
- Ensure models are in `public/models/`
- Check browser console for 404 errors

### "Camera not working"
- Grant camera permissions
- Use HTTPS or localhost
- Check browser compatibility

### "Slow inference"
- Switch to WebGPU backend in code
- Reduce input size
- Use fewer classes

## License

Apache 2.0
