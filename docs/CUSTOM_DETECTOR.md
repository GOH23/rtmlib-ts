# CustomDetector API

Maximum flexibility detector for any ONNX model with customizable preprocessing and postprocessing.

## Overview

`CustomDetector` provides a low-level, flexible API for running inference with any ONNX model. It supports automatic preprocessing, custom preprocessing/postprocessing functions, and works with various input sources (canvas, video, image, file, blob).

## Installation

```bash
npm install rtmlib-ts
```

## Quick Start

### Basic Usage

```typescript
import { CustomDetector } from 'rtmlib-ts';

// Initialize with model path
const detector = new CustomDetector({
  model: 'path/to/model.onnx',
});

await detector.init();

// Run inference
const result = await detector.runFromCanvas(canvas);
console.log(result.outputs); // Raw ONNX outputs
console.log(result.data);    // Processed data
console.log(result.inferenceTime); // Inference time in ms
```

### Simple Classification

```typescript
const detector = new CustomDetector({
  model: 'https://example.com/mobilenet.onnx',
  inputSize: [224, 224],
  normalization: {
    mean: [123.675, 116.28, 103.53],
    std: [58.395, 57.12, 57.375],
  },
});

await detector.init();

const result = await detector.runFromCanvas(canvas);
const output = detector.getOutputTensor(result.outputs);
const predictedClass = output.data.indexOf(Math.max(...output.data));
console.log(`Predicted class: ${predictedClass}`);
```

### Custom Preprocessing and Postprocessing

```typescript
const detector = new CustomDetector({
  model: 'path/to/model.onnx',
  inputSize: [512, 512],
  preprocessing: (imageData, config) => {
    // Custom preprocessing logic
    const tensor = new Float32Array(3 * 512 * 512);
    // ... your preprocessing
    return tensor;
  },
  postprocessing: (outputs, metadata) => {
    // Custom postprocessing logic
    const output = outputs['output'];
    return {
      boxes: decodeBoxes(output),
      scores: decodeScores(output),
    };
  },
});
```

## API Reference

### Constructor

```typescript
new CustomDetector(config: CustomDetectorConfig)
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | `string` | required | Path to ONNX model |
| `inputName` | `string` | auto | Input tensor name |
| `outputNames` | `string[]` | auto | Output tensor names |
| `inputShape` | `[number, number, number, number]` | `[1, 3, 224, 224]` | Expected input shape |
| `inputSize` | `[number, number]` | optional | Input size for automatic preprocessing |
| `preprocessing` | `function` | auto | Custom preprocessing function |
| `postprocessing` | `function` | auto | Custom postprocessing function |
| `normalization` | `object` | `{ mean: [0,0,0], std: [1,1,1] }` | Normalization parameters |
| `keepAspectRatio` | `boolean` | `true` | Keep aspect ratio during preprocessing |
| `backgroundColor` | `string` | `'#000000'` | Background color for letterbox |
| `backend` | `'wasm' \| 'webgpu'` | `'wasm'` | Execution backend |
| `cache` | `boolean` | `true` | Enable model caching |
| `metadata` | `any` | optional | Custom metadata for postprocessing |

### Methods

#### `init()`

Initialize the model.

```typescript
await detector.init();
```

#### `runFromCanvas()`

Run inference on HTMLCanvasElement.

```typescript
async runFromCanvas<T = any>(
  canvas: HTMLCanvasElement
): Promise<DetectionResult<T>>
```

#### `runFromVideo()`

Run inference on HTMLVideoElement.

```typescript
async runFromVideo<T = any>(
  video: HTMLVideoElement,
  targetCanvas?: HTMLCanvasElement
): Promise<DetectionResult<T>>
```

#### `runFromImage()`

Run inference on HTMLImageElement.

```typescript
async runFromImage<T = any>(
  image: HTMLImageElement,
  targetCanvas?: HTMLCanvasElement
): Promise<DetectionResult<T>>
```

#### `runFromBitmap()`

Run inference on ImageBitmap.

```typescript
async runFromBitmap<T = any>(
  bitmap: ImageBitmap,
  targetCanvas?: HTMLCanvasElement
): Promise<DetectionResult<T>>
```

#### `runFromFile()`

Run inference on File object.

```typescript
async runFromFile<T = any>(
  file: File,
  targetCanvas?: HTMLCanvasElement
): Promise<DetectionResult<T>>
```

#### `runFromBlob()`

Run inference on Blob.

```typescript
async runFromBlob<T = any>(
  blob: Blob,
  targetCanvas?: HTMLCanvasElement
): Promise<DetectionResult<T>>
```

#### `run()`

Low-level method for raw image data with custom preprocessing.

```typescript
async run<T = any>(
  imageData: ImageData,
  width: number,
  height: number,
  metadata?: any
): Promise<DetectionResult<T>>
```

#### `getModelInfo()`

Get model input/output information.

```typescript
const info = detector.getModelInfo();
console.log(`Inputs: ${info.inputNames}`);
console.log(`Outputs: ${info.outputNames}`);
```

#### `getOutputTensor()`

Get tensor by name from outputs.

```typescript
const tensor = detector.getOutputTensor<ort.Tensor>(
  result.outputs,
  'output_name'  // optional, uses first output if not specified
);
```

#### `dispose()`

Release resources.

```typescript
detector.dispose();
```

### Types

#### `DetectionResult<T>`

```typescript
interface DetectionResult<T = any> {
  outputs: Record<string, ort.Tensor>;  // Raw model outputs
  data: T;                               // Processed results
  inferenceTime: number;                 // Inference time in ms
  inputShape: number[];                  // Input shape used
}
```

#### `CustomDetectorConfig`

```typescript
interface CustomDetectorConfig {
  model: string;
  inputName?: string;
  outputNames?: string[];
  inputShape?: [number, number, number, number];
  preprocessing?: (data: ImageData, config: CustomDetectorConfig) => Float32Array | ort.Tensor;
  postprocessing?: (outputs: Record<string, ort.Tensor>, metadata: any) => any;
  backend?: 'wasm' | 'webgpu';
  cache?: boolean;
  metadata?: any;
  normalization?: {
    mean: number[];
    std: number[];
  };
  inputSize?: [number, number];
  keepAspectRatio?: boolean;
  backgroundColor?: string;
}
```

## Examples

### Image Classification

```typescript
import { CustomDetector } from 'rtmlib-ts';

const detector = new CustomDetector({
  model: 'https://example.com/resnet50.onnx',
  inputSize: [224, 224],
  normalization: {
    mean: [123.675, 116.28, 103.53],
    std: [58.395, 57.12, 57.375],
  },
  postprocessing: (outputs) => {
    const output = outputs['output'];
    const scores = Array.from(output.data as Float32Array);
    const predictedClass = scores.indexOf(Math.max(...scores));
    const confidence = scores[predictedClass];
    return { predictedClass, confidence, scores };
  },
});

await detector.init();

const result = await detector.runFromCanvas(canvas);
console.log(`Predicted: ${result.data.predictedClass} (${result.data.confidence * 100}%)`);
```

### Object Detection (YOLO-style)

```typescript
const detector = new CustomDetector({
  model: 'path/to/yolo.onnx',
  inputSize: [640, 640],
  normalization: {
    mean: [0, 0, 0],
    std: [1, 1, 1],
  },
  postprocessing: (outputs, metadata) => {
    const output = outputs['output'];
    const data = output.data as Float32Array;
    const numDetections = output.dims[1];
    
    const detections = [];
    for (let i = 0; i < numDetections; i++) {
      const idx = i * 6;
      const [x1, y1, x2, y2, conf, classId] = [
        data[idx],
        data[idx + 1],
        data[idx + 2],
        data[idx + 3],
        data[idx + 4],
        Math.round(data[idx + 5]),
      ];
      
      if (conf > 0.5) {
        detections.push({
          bbox: { x1, y1, x2, y2 },
          classId,
          confidence: conf,
        });
      }
    }
    
    return detections;
  },
});

await detector.init();
const result = await detector.runFromCanvas(canvas);
console.log(`Detected ${result.data.length} objects`);
```

### Semantic Segmentation

```typescript
const detector = new CustomDetector({
  model: 'path/to/deeplab.onnx',
  inputSize: [512, 512],
  normalization: {
    mean: [123.675, 116.28, 103.53],
    std: [58.395, 57.12, 57.375],
  },
  postprocessing: (outputs) => {
    const mask = outputs['mask'];
    const data = mask.data as Float32Array;
    const [batch, numClasses, height, width] = mask.dims;
    
    // Get class for each pixel
    const segmentation = new Uint8Array(height * width);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIdx = y * width + x;
        let maxClass = 0;
        let maxScore = -Infinity;
        
        for (let c = 0; c < numClasses; c++) {
          const score = data[c * height * width + pixelIdx];
          if (score > maxScore) {
            maxScore = score;
            maxClass = c;
          }
        }
        
        segmentation[pixelIdx] = maxClass;
      }
    }
    
    return { segmentation, height, width };
  },
});

await detector.init();
const result = await detector.runFromCanvas(canvas);
console.log(`Segmentation map: ${result.data.height}x${result.data.width}`);
```

### Face Landmarks

```typescript
const detector = new CustomDetector({
  model: 'path/to/face_landmarks.onnx',
  inputSize: [192, 192],
  preprocessing: (imageData, config) => {
    // Custom face preprocessing
    const tensor = new Float32Array(3 * 192 * 192);
    const { data, width, height } = imageData;
    
    for (let i = 0; i < data.length; i += 4) {
      const pixelIdx = i / 4;
      tensor[pixelIdx] = (data[i] - 127.5) / 127.5;
      tensor[pixelIdx + width * height] = (data[i + 1] - 127.5) / 127.5;
      tensor[pixelIdx + 2 * width * height] = (data[i + 2] - 127.5) / 127.5;
    }
    
    return tensor;
  },
  postprocessing: (outputs) => {
    const landmarks = outputs['landmarks'];
    const data = landmarks.data as Float32Array;
    const numLandmarks = landmarks.dims[1];
    
    const points = [];
    for (let i = 0; i < numLandmarks; i++) {
      points.push({
        x: data[i * 2],
        y: data[i * 2 + 1],
      });
    }
    
    return points;
  },
});

await detector.init();
const result = await detector.runFromCanvas(canvas);
console.log(`Detected ${result.data.length} facial landmarks`);
```

## Preprocessing Options

### Automatic Preprocessing

If you provide `inputSize`, automatic preprocessing with letterbox is applied:

```typescript
const detector = new CustomDetector({
  model: 'path/to/model.onnx',
  inputSize: [224, 224],
  keepAspectRatio: true,  // Letterbox padding
  backgroundColor: '#000000',
  normalization: {
    mean: [123.675, 116.28, 103.53],
    std: [58.395, 57.12, 57.375],
  },
});
```

### Custom Preprocessing Function

```typescript
const detector = new CustomDetector({
  model: 'path/to/model.onnx',
  preprocessing: (imageData, config) => {
    const { data, width, height } = imageData;
    const tensor = new Float32Array(3 * width * height);
    
    // Custom preprocessing logic
    for (let i = 0; i < data.length; i += 4) {
      const pixelIdx = i / 4;
      tensor[pixelIdx] = data[i] / 255;  // R
      tensor[pixelIdx + width * height] = data[i + 1] / 255;  // G
      tensor[pixelIdx + 2 * width * height] = data[i + 2] / 255;  // B
    }
    
    return tensor;
  },
});
```

## Postprocessing Options

### Automatic Postprocessing

If no `postprocessing` function is provided, raw ONNX outputs are returned:

```typescript
const result = await detector.runFromCanvas(canvas);
console.log(result.outputs); // Record<string, ort.Tensor>
```

### Custom Postprocessing Function

```typescript
const detector = new CustomDetector({
  model: 'path/to/model.onnx',
  postprocessing: (outputs, metadata) => {
    // Process outputs
    const output1 = outputs['output1'];
    const output2 = outputs['output2'];
    
    // Your custom logic
    return {
      processed1: processOutput1(output1),
      processed2: processOutput2(output2),
    };
  },
});
```

## Performance Optimization

### 1. Use WebGPU Backend

```typescript
const detector = new CustomDetector({
  model: 'path/to/model.onnx',
  backend: 'webgpu',  // Faster than WASM
});
```

### 2. Enable Model Caching

```typescript
const detector = new CustomDetector({
  model: 'path/to/model.onnx',
  cache: true,  // Cache model for faster subsequent loads
});
```

### 3. Reuse Detector Instance

```typescript
const detector = new CustomDetector({ model: 'path/to/model.onnx' });
await detector.init();

// Reuse for multiple inferences
for (const frame of frames) {
  const result = await detector.run(frame.data, frame.width, frame.height);
}
```

### 4. Pre-allocate Resources

```typescript
// Create canvas once
const canvas = document.createElement('canvas');
canvas.width = 640;
canvas.height = 480;

// Reuse for all inferences
const result = await detector.runFromCanvas(canvas);
```

## Browser Support

| Browser | Version | Backend |
|---------|---------|---------|
| Chrome | 94+ | WASM, WebGPU |
| Edge | 94+ | WASM, WebGPU |
| Firefox | 95+ | WASM |
| Safari | 16.4+ | WASM |

## Troubleshooting

### "Model loading failed"

- Ensure model is accessible via HTTP (not `file://` protocol)
- Use a local server: `python -m http.server 8080`
- Check CORS headers

### "Input name not found"

- Use `getModelInfo()` to check available input names
- Specify `inputName` in config if auto-detection fails

### "Output shape mismatch"

- Check model's expected input shape with `getModelInfo()`
- Adjust `inputSize` or `inputShape` in config

### "Slow inference"

- Switch to WebGPU backend if available
- Reduce `inputSize`
- Use model quantization (INT8 models)

## License

Apache 2.0
