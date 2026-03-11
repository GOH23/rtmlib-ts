# YOLO12 Person Detection - Web Demo

Real-time person detection using YOLO12 and ONNX Runtime Web.

## Quick Start

### Option 1: Using Python HTTP Server

```bash
# From the rtmlib-ts directory
python -m http.server 8080 --directory examples

# Open in browser
http://localhost:8080/index.html
```

### Option 2: Using Node.js (http-server)

```bash
# Install http-server globally
npm install -g http-server

# Run server
http-server examples -p 8080

# Open in browser
http://localhost:8080/index.html
```

### Option 3: Using VS Code Live Server

1. Install "Live Server" extension in VS Code
2. Right-click on `examples/index.html`
3. Select "Open with Live Server"

## Features

- 🎯 **Person Detection**: Detects people in images using YOLO12n model
- 🚀 **Fast Inference**: Runs entirely in the browser using WebAssembly
- 📊 **Real-time Stats**: Shows detection count and inference time
- 🎨 **Visual Results**: Bounding boxes with confidence scores
- ⚙️ **Adjustable Threshold**: Control detection confidence

## How It Works

1. **Model Loading**: The YOLO12n ONNX model is loaded via fetch
2. **Image Upload**: User uploads an image via drag-drop or file picker
3. **Preprocessing**: Image is resized with letterbox padding (black background)
4. **Inference**: ONNX Runtime Web runs the model using WebAssembly
5. **Postprocessing**: Filter by confidence, transform coordinates, apply NMS
6. **Visualization**: Draw bounding boxes on canvas

## Model

- **Name**: YOLO12n (Nano)
- **Input Size**: 640x640
- **Classes**: 80 COCO classes (we filter for class 0 = person)
- **Size**: ~11 MB
- **Source**: Ultralytics

## Browser Support

- Chrome 94+ (WebAssembly SIMD)
- Firefox 95+
- Safari 16.4+
- Edge 94+

## File Structure

```
examples/
├── index.html          # Main web demo page
├── models/
│   └── yolov12n.onnx   # YOLO12 model
└── 8.png               # Sample image
```

## API Usage (TypeScript)

```typescript
import { YOLO12 } from './dist/index.js';

// Initialize detector
const detector = new YOLO12(
  'models/yolov12n.onnx',
  [640, 640],  // input size
  0.45,        // NMS threshold
  0.5          // confidence threshold
);

await detector.init();

// Detect people in image
const imageData = ...; // Uint8Array RGB image
const width = 640;
const height = 480;

const detections = await detector.call(imageData, width, height);

// detections: Detection[]
// Each detection has: bbox, score, classId
```

## Troubleshooting

### Model fails to load
- Ensure the server is running (not just opening file://)
- Check browser console for CORS errors
- Verify model file exists in `examples/models/`

### Slow inference
- First run includes WASM compilation (subsequent runs are faster)
- Use Chrome/Edge for best WebAssembly performance
- Reduce image size for faster processing

### No detections
- Lower the confidence threshold
- Ensure person is visible in the image
- Check that the image format is supported (PNG, JPG, WebP)
