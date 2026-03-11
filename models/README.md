# Models

This directory contains pre-trained ONNX models for object detection and pose estimation.

## Directory Structure

```
models/
├── yolo/
│   ├── yolov12n.onnx      # YOLO12 Nano (object detection)
│   └── yolov26n.onnx      # YOLO12 Nano (alternative)
├── rtmpose/
│   └── end2end.onnx       # RTMW (pose estimation)
└── README.md
```

## Available Models

### Object Detection

| Model | File | Size | Classes | Input | Description |
|-------|------|------|---------|-------|-------------|
| YOLO12n | `yolo/yolov12n.onnx` | ~11 MB | 80 COCO | 640×640 | YOLO12 Nano - fastest |
| YOLOv26n | `yolo/yolov26n.onnx` | ~11 MB | 80 COCO | 640×640 | YOLO12 Nano variant - balanced |

**Model Selection:**
- **YOLO12n**: Fastest inference, best for real-time applications
- **YOLv26n**: Slightly different architecture, may have better accuracy on some objects

### Pose Estimation

| Model | File | Size | Keypoints | Input | Description |
|-------|------|------|-----------|-------|-------------|
| RTMW | `rtmpose/end2end.onnx` | ~50 MB | 17 COCO | 384×288 | RTMW wholebody |

## Usage

### ObjectDetector

```typescript
import { ObjectDetector } from 'rtmlib-ts';

const detector = new ObjectDetector({
  model: 'models/yolo/yolov12n.onnx',
  classes: ['person', 'car'],  // Filter classes or null for all
});
await detector.init();

const objects = await detector.detectFromCanvas(canvas);
```

### PoseDetector

```typescript
import { PoseDetector } from 'rtmlib-ts';

const detector = new PoseDetector({
  detModel: 'models/yolo/yolov12n.onnx',
  poseModel: 'models/rtmpose/end2end.onnx',
});
await detector.init();

const people = await detector.detectFromCanvas(canvas);
```

## Model Paths (Relative)

When using the library, reference models with relative paths from your web root:

```typescript
// From web root (rtmlib-ts/)
const detector = new ObjectDetector({
  model: './models/yolo/yolov12n.onnx',
});

// From examples/
const detector = new ObjectDetector({
  model: '../models/yolo/yolov12n.onnx',
});
```

## Performance

### YOLO12n (Object Detection)

| Backend | Input Size | Inference Time |
|---------|------------|----------------|
| WASM | 640×640 | ~80ms |
| WASM | 416×416 | ~40ms |
| WebGPU | 640×640 | ~30ms |

### RTMW (Pose Estimation)

| Backend | Input Size | Inference Time (per person) |
|---------|------------|---------------------------|
| WASM | 384×288 | ~25ms |
| WebGPU | 384×288 | ~10ms |

## COCO Classes (80)

YOLO12n detects all 80 COCO classes:

```
0: person         20: elephant      40: cup           60: toilet
1: bicycle        21: bear          41: fork          61: tv
2: car            22: zebra         42: knife         62: laptop
3: motorcycle     23: giraffe       43: spoon         63: mouse
4: airplane       24: backpack      44: bowl          64: remote
5: bus            25: umbrella      45: banana        65: keyboard
6: train          26: handbag       46: apple         66: cell phone
7: truck          27: tie           47: sandwich      67: microwave
8: boat           28: suitcase      48: orange        68: oven
9: traffic light  29: frisbee       49: broccoli      69: toaster
10: fire hydrant  30: skis          50: carrot        70: sink
11: stop sign     31: snowboard     51: hot dog       71: refrigerator
12: parking meter 32: sports ball   52: pizza         72: book
13: bench         33: kite          53: donut         73: clock
14: bird          34: baseball bat  54: cake          74: vase
15: cat           35: baseball glove 55: chair        75: scissors
16: dog           36: skateboard    56: couch         76: teddy bear
17: horse         37: surfboard     57: potted plant  77: hair drier
18: sheep         38: tennis racket 58: bed           78: toothbrush
19: cow           39: bottle        59: dining table
```

## License

- YOLO12: AGPL-3.0 (Ultralytics)
- RTMW: Apache-2.0 (OpenMMLab)
