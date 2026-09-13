/**
 * ObjectDetector - Universal object detection API
 * Supports YOLO (Ultralytics ONNX export) and MediaPipe Tasks Vision
 * (EfficientDet-Lite0) for multi-class detection. The default backend is
 * `'yolo'` (YOLO12n on HuggingFace). Override with `detectorType:
 * 'mediapipe'` to switch backends.
 *
 * @example
 * ```typescript
 * // Default — YOLO12n on HuggingFace, person-only
 * const detector = new ObjectDetector({
 *   classes: ['person'],
 * });
 * await detector.init();
 *
 * // Or with MediaPipe EfficientDet-Lite0
 * const detector = new ObjectDetector({
 *   detectorType: 'mediapipe',
 *   classes: ['person'],
 * });
 * await detector.init();
 *
 * // Detect from canvas
 * const objects = await detector.detectFromCanvas(canvas);
 * ```
 */

import * as ort from 'onnxruntime-web/all';
import { MediaPipeObjectDetector, MediaPipeDetectedObject } from './mediaPipeObjectDetector';
import { initOnnxRuntimeWeb } from '../core/onnxRuntime';
import { resolveYoloModelUrl, type YoloVersion } from '../models/yoloModels';
import { attachStats } from '../core/stats';
import { loadBitmapFromBlob, loadImageFromFile } from '../core/sourceLoaders';
import { letterboxToCanvas } from '../core/preprocessing';
import { loadOnnxSession } from '../core/onnxSession';
import { createLogger } from '../core/logger';
import type { WebNNProviderOptionsOrUndefined } from '../types/index';

// Configure ONNX Runtime Web (only in browser environment)
initOnnxRuntimeWeb();

const log = createLogger('ObjectDetector');

/**
 * COCO 80-class names
 */
export const COCO_CLASSES: string[] = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
];

/**
 * Backend type for ObjectDetector
 */
export type ObjectDetectorBackend = 'yolo' | 'mediapipe';

/**
 * Configuration options for ObjectDetector
 */
export interface ObjectDetectorConfig {
  /** Path to YOLO detection model (optional - uses default YOLOv12n from HuggingFace if not specified) */
  model?: string;
  /** YOLO version shortcut. If `model` is also given, the URL wins; if
   * `model` is omitted, this resolves to a default ONNX URL via
   * `YOLO_VERSIONS[version]`. Supports `'yolov8n'`, `'yolov12n'`,
   * `'yolo26n'`. Default `'yolov12n'`. */
  yoloVersion?: YoloVersion;
  /** Input size (default: [416, 416] for speed) */
  inputSize?: [number, number];
  /** Confidence threshold (default: 0.5) */
  confidence?: number;
  /** NMS IoU threshold (default: 0.45) */
  nmsThreshold?: number;
  /** Classes to detect (null = all, default: ['person']) */
  classes?: string[] | null;
  /** Execution backend (default: 'webgl' for best compatibility) */
  backend?: 'wasm' | 'webgl' | 'webgpu' | 'webnn';
  /** WebNN provider options (only used when backend is 'webnn') */
  webnnOptions?: WebNNProviderOptionsOrUndefined;
  /** Device type for WebNN/WebGPU (default: 'gpu' for high performance) */
  deviceType?: 'cpu' | 'gpu' | 'npu';
  /** Power preference for WebNN/WebGPU (default: 'high-performance') */
  powerPreference?: 'default' | 'low-power' | 'high-performance';
  /** Performance mode (default: 'balanced') */
  mode?: 'performance' | 'balanced' | 'lightweight';
  /** Device type (for future use) */
  device?: 'cpu' | 'gpu';
  /** Enable model caching (default: true) */
  cache?: boolean;
  /** Backend type: 'yolo' for ONNX models, 'mediapipe' for MediaPipe Tasks Vision (default: 'yolo') */
  detectorType?: ObjectDetectorBackend;
  /** MediaPipe model path (only used when detectorType is 'mediapipe') */
  mediaPipeModelPath?: string;
  /** MediaPipe score threshold (only used when detectorType is 'mediapipe') */
  mediaPipeScoreThreshold?: number;
  /** MediaPipe max results (only used when detectorType is 'mediapipe') */
  mediaPipeMaxResults?: number;
}

/**
 * Detected object with bounding box and class
 */
export interface DetectedObject {
  /** Bounding box coordinates */
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
  };
  /** Class ID (0-79 for COCO) */
  classId: number;
  /** Class name */
  className: string;
  /** Detection confidence (0-1) */
  confidence: number;
}

/**
 * Detection statistics
 */
export interface DetectionStats {
  /** Total number of detections */
  totalCount: number;
  /** Detections per class */
  classCounts: Record<string, number>;
  /** Inference time (ms) */
  inferenceTime: number;
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: Omit<Required<ObjectDetectorConfig>, 'webnnOptions'> & {
  webnnOptions?: WebNNProviderOptionsOrUndefined;
} = {
  model: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  yoloVersion: 'yolov12n',
  inputSize: [416, 416],  // Faster default
  confidence: 0.5,
  nmsThreshold: 0.45,
  classes: ['person'],
  backend: 'webgl',  // Default to WebGL for best compatibility
  webnnOptions: undefined as WebNNProviderOptionsOrUndefined,
  deviceType: 'gpu',
  powerPreference: 'high-performance',
  mode: 'balanced',
  device: 'cpu',
  cache: true,
  detectorType: 'yolo',
  mediaPipeModelPath: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite',
  mediaPipeScoreThreshold: 0.5,
  mediaPipeMaxResults: -1,
};


// Performance presets
const MODE_PRESETS: Record<string, { inputSize: [number, number]; confidence: number }> = {
  performance: { inputSize: [640, 640], confidence: 0.3 },  // High accuracy
  balanced: { inputSize: [416, 416], confidence: 0.5 },     // Balanced
  lightweight: { inputSize: [320, 320], confidence: 0.6 },  // Fastest
};

export class ObjectDetector {
  private config: Omit<Required<ObjectDetectorConfig>, 'webnnOptions'> & {
    webnnOptions?: WebNNProviderOptionsOrUndefined;
  };
  private session: ort.InferenceSession | null = null;
  private mediaPipeDetector: MediaPipeObjectDetector | null = null;
  private initialized = false;
  private classFilter: Set<number> | null = null;

  // Pre-allocated reusable resources for performance
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private tensorBuffer: Float32Array | null = null;
  private inputSize: [number, number] = [416, 416];

  constructor(config: ObjectDetectorConfig) {
    // Apply mode preset if specified
    let finalConfig = { ...DEFAULT_CONFIG, ...config };

    // If the caller specified `yoloVersion` but not `model`, resolve to the
    // version's default URL. `model` (raw URL) always wins over `yoloVersion`.
    if (!config.model && config.yoloVersion) {
      finalConfig.model = resolveYoloModelUrl(config.yoloVersion);
    }

    // Apply mode preset if specified
    if (config.mode && MODE_PRESETS[config.mode]) {
      const preset = MODE_PRESETS[config.mode];
      // Only override if not explicitly set
      if (!config.inputSize) finalConfig.inputSize = preset.inputSize;
      if (!config.confidence) finalConfig.confidence = preset.confidence;
    }

    this.config = finalConfig;
    this.updateClassFilter();

    log.log(`Initialized with mode: ${config.mode || 'balanced'}, input: ${this.config.inputSize[0]}x${this.config.inputSize[1]}, detectorType: ${this.config.detectorType}`);
  }

  /**
   * Update class filter based on config
   */
  private updateClassFilter(): void {
    if (!this.config.classes) {
      this.classFilter = null;
      return;
    }

    this.classFilter = new Set<number>();
    this.config.classes.forEach((className) => {
      const classId = COCO_CLASSES.indexOf(className.toLowerCase());
      if (classId !== -1) {
        this.classFilter!.add(classId);
      } else {
        log.warn(`Unknown class: ${className}`);
      }
    });
  }

  /**
   * Set which classes to detect
   * @param classes - Array of class names or null for all classes
   */
  setClasses(classes: string[] | null): void {
    this.config.classes = classes;
    this.updateClassFilter();
  }

  /**
   * Get list of available classes. Both `'yolo'` and `'mediapipe'`
   * backends use the same COCO 80-class list.
   */
  getAvailableClasses(): string[] {
    return [...COCO_CLASSES];
  }

  /**
   * Get currently filtered classes
   */
  getFilteredClasses(): string[] | null {
    return this.config.classes;
  }

  /**
   * Initialize detection model and pre-allocate resources
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    try {
      // Initialize based on detector type. Order matters: each branch
      // takes full ownership of its own session / native handle.
      if (this.config.detectorType === 'mediapipe') {
        await this.initMediaPipe();
      } else {
        await this.initYOLO();
      }
    } catch (error) {
      log.error('❌ Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Initialize YOLO model (original implementation)
   */
  private async initYOLO(): Promise<void> {
    this.session = await loadOnnxSession({
      url: this.config.model,
      backend: this.config.backend,
      cache: this.config.cache,
      deviceType: this.config.deviceType,
      powerPreference: this.config.powerPreference,
      fallbackProviders: ['wasm'],
      logPrefix: '[ObjectDetector]',
    });

    // Pre-allocate canvas and tensor buffer for performance
    const [w, h] = this.config.inputSize;
    this.inputSize = [w, h];

    this.canvas = document.createElement('canvas');
    this.canvas.width = w;
    this.canvas.height = h;
    this.ctx = this.canvas.getContext('2d', {
      willReadFrequently: true,
      alpha: false
    })!;

    // Pre-allocate tensor buffer (3 channels * width * height)
    this.tensorBuffer = new Float32Array(3 * w * h);

    this.initialized = true;
  }

  /**
   * Initialize MediaPipe detector
   */
  private async initMediaPipe(): Promise<void> {
    log.log(`Initializing MediaPipe detector from: ${this.config.mediaPipeModelPath}`);

    this.mediaPipeDetector = new MediaPipeObjectDetector({
      modelPath: this.config.mediaPipeModelPath,
      scoreThreshold: this.config.mediaPipeScoreThreshold,
      maxResults: this.config.mediaPipeMaxResults,
      categoryAllowlist: this.config.classes || undefined,
    });

    await this.mediaPipeDetector.init();

    this.initialized = true;
    log.log('✅ MediaPipe Initialized');
  }

  /**
   * Detect objects from HTMLCanvasElement
   */
  async detectFromCanvas(canvas: HTMLCanvasElement): Promise<DetectedObject[]> {
    if (!this.initialized) {
      await this.init();
    }

    // Use MediaPipe if selected
    if (this.config.detectorType === 'mediapipe' && this.mediaPipeDetector) {
      const mpDetections = await this.mediaPipeDetector.detectFromCanvas(canvas);
      return this.convertMediaPipeDetections(mpDetections, canvas.width, canvas.height);
    }

    // Otherwise use YOLO
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context from canvas');
    }

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return this.detect(new Uint8Array(imageData.data.buffer), canvas.width, canvas.height);
  }

  /**
   * Detect objects from HTMLVideoElement
   */
  async detectFromVideo(
    video: HTMLVideoElement,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectedObject[]> {
    if (!this.initialized) {
      await this.init();
    }

    // Use MediaPipe if selected
    if (this.config.detectorType === 'mediapipe' && this.mediaPipeDetector) {
      const mpDetections = await this.mediaPipeDetector.detectFromVideo(video);
      return this.convertMediaPipeDetections(mpDetections, video.videoWidth, video.videoHeight);
    }

    // Otherwise use YOLO
    if (video.readyState < 2) {
      throw new Error('Video not ready. Ensure video is loaded and playing.');
    }

    const canvas = targetCanvas || document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context from canvas');
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    return this.detect(new Uint8Array(imageData.data.buffer), canvas.width, canvas.height);
  }

  /**
   * Detect objects from HTMLImageElement
   */
  async detectFromImage(
    image: HTMLImageElement,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectedObject[]> {
    if (!this.initialized) {
      await this.init();
    }

    // Use MediaPipe if selected
    if (this.config.detectorType === 'mediapipe' && this.mediaPipeDetector) {
      const mpDetections = await this.mediaPipeDetector.detectFromImage(image);
      return this.convertMediaPipeDetections(mpDetections, image.naturalWidth, image.naturalHeight);
    }

    // Otherwise use YOLO
    if (!image.complete || !image.naturalWidth) {
      throw new Error('Image not loaded. Ensure image is fully loaded.');
    }

    const canvas = targetCanvas || document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context from canvas');
    }

    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    return this.detect(new Uint8Array(imageData.data.buffer), canvas.width, canvas.height);
  }

  /**
   * Detect objects from ImageBitmap
   */
  async detectFromBitmap(
    bitmap: ImageBitmap,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectedObject[]> {
    if (!this.initialized) {
      await this.init();
    }

    // Use MediaPipe if selected
    if (this.config.detectorType === 'mediapipe' && this.mediaPipeDetector) {
      const mpDetections = await this.mediaPipeDetector.detectFromBitmap(bitmap);
      return this.convertMediaPipeDetections(mpDetections, bitmap.width, bitmap.height);
    }

    // Otherwise use YOLO
    const canvas = targetCanvas || document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context from canvas');
    }

    ctx.drawImage(bitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    return this.detect(new Uint8Array(imageData.data.buffer), canvas.width, canvas.height);
  }

  /**
   * Detect objects from File
   */
  async detectFromFile(
    file: File,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectedObject[]> {
    if (!this.initialized) {
      await this.init();
    }

    // Use MediaPipe if selected
    if (this.config.detectorType === 'mediapipe' && this.mediaPipeDetector) {
      const mpDetections = await this.mediaPipeDetector.detectFromFile(file);
      const img = await loadImageFromFile(file);
      return this.convertMediaPipeDetections(mpDetections, img.width, img.height);
    }

    // YOLO goes through the same image-load → detectFromImage path.
    const img = await loadImageFromFile(file);
    return this.detectFromImage(img, targetCanvas);
  }

  /**
   * Detect objects from Blob
   */
  async detectFromBlob(
    blob: Blob,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectedObject[]> {
    if (!this.initialized) {
      await this.init();
    }

    // Use MediaPipe if selected
    if (this.config.detectorType === 'mediapipe' && this.mediaPipeDetector) {
      const mpDetections = await this.mediaPipeDetector.detectFromBlob(blob);
      const bitmap = await loadBitmapFromBlob(blob);
      try {
        return this.convertMediaPipeDetections(mpDetections, bitmap.width, bitmap.height);
      } finally {
        bitmap.close();
      }
    }

    // YOLO goes through the same bitmap pipeline.
    const bitmap = await loadBitmapFromBlob(blob);
    try {
      return await this.detectFromBitmap(bitmap, targetCanvas);
    } finally {
      bitmap.close();
    }
  }

  /**
   * Convert MediaPipe detections to standard format
   */
  private convertMediaPipeDetections(
    mpDetections: MediaPipeDetectedObject[],
    width: number,
    height: number
  ): DetectedObject[] {
    return mpDetections.map(mpDet => {
      const classId = COCO_CLASSES.indexOf(mpDet.categoryName.toLowerCase());
      return {
        bbox: {
          x1: mpDet.bbox.x1,
          y1: mpDet.bbox.y1,
          x2: mpDet.bbox.x1 + mpDet.bbox.width,
          y2: mpDet.bbox.y1 + mpDet.bbox.height,
          confidence: mpDet.score,
        },
        classId: classId !== -1 ? classId : mpDet.categoryId,
        className: mpDet.categoryName,
        confidence: mpDet.score,
      };
    });
  }

  /**
   * Detect objects from raw image data
   */
  async detect(
    imageData: Uint8Array,
    width: number,
    height: number
  ): Promise<DetectedObject[]> {
    if (!this.initialized) {
      await this.init();
    }

    const startTime = performance.now();

    const [inputH, inputW] = this.config.inputSize;

    // Preprocess
    const { tensor, paddingX, paddingY, scaleX, scaleY } = this.preprocess(
      imageData,
      width,
      height,
      [inputW, inputH]
    );

    // Inference - use dynamic input name
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, inputH, inputW]);
    const inputName = this.session!.inputNames[0];  // Dynamic: 'images' or 'pixel_values'

    const feeds: Record<string, ort.Tensor> = {};
    feeds[inputName] = inputTensor;

    const results = await this.session!.run(feeds);
    const output = results[this.session!.outputNames[0]];

    // Postprocess
    const detections = this.postprocess(
      output.data as Float32Array,
      output.dims[1],
      output.dims as number[],
      width,
      height,
      paddingX,
      paddingY,
      scaleX,
      scaleY
    );

    const inferenceTime = performance.now() - startTime;

    // Attach stats
    attachStats(detections, this.calculateStats(detections, inferenceTime));

    return detections;
  }

  /**
   * Optimized preprocess with resource reuse
   */
  private preprocess(
    imageData: Uint8Array,
    imgWidth: number,
    imgHeight: number,
    inputSize: [number, number]
  ): {
    tensor: Float32Array;
    paddingX: number;
    paddingY: number;
    scaleX: number;
    scaleY: number;
  } {
    const [inputW, inputH] = inputSize;

    // Reuse pre-allocated canvas
    if (!this.canvas || !this.ctx) {
      this.canvas = document.createElement('canvas');
      this.canvas.width = inputW;
      this.canvas.height = inputH;
      this.ctx = this.canvas.getContext('2d', {
        willReadFrequently: true,
        alpha: false
      })!;
      this.tensorBuffer = new Float32Array(3 * inputW * inputH);
    }

    const ctx = this.ctx;

    // Letterbox into the pre-allocated target canvas.
    const meta = letterboxToCanvas(imageData, imgWidth, imgHeight, inputW, inputH, ctx);

    const paddedData = ctx.getImageData(0, 0, inputW, inputH);

    // Optimized normalization loop (reuse buffer)
    const tensor = this.tensorBuffer!;
    const len = paddedData.data.length;
    const planeSize = inputW * inputH;

    // Unroll loop for speed (process 4 pixels at once)
    for (let i = 0; i < len; i += 16) {
      const i1 = i, i2 = i + 4, i3 = i + 8, i4 = i + 12;
      const p1 = i1 / 4, p2 = i2 / 4, p3 = i3 / 4, p4 = i4 / 4;

      // R channel
      tensor[p1] = paddedData.data[i1] * 0.003921569;  // / 255
      tensor[p2] = paddedData.data[i2] * 0.003921569;
      tensor[p3] = paddedData.data[i3] * 0.003921569;
      tensor[p4] = paddedData.data[i4] * 0.003921569;

      // G channel
      tensor[p1 + planeSize] = paddedData.data[i1 + 1] * 0.003921569;
      tensor[p2 + planeSize] = paddedData.data[i2 + 1] * 0.003921569;
      tensor[p3 + planeSize] = paddedData.data[i3 + 1] * 0.003921569;
      tensor[p4 + planeSize] = paddedData.data[i4 + 1] * 0.003921569;

      // B channel
      tensor[p1 + planeSize * 2] = paddedData.data[i1 + 2] * 0.003921569;
      tensor[p2 + planeSize * 2] = paddedData.data[i2 + 2] * 0.003921569;
      tensor[p3 + planeSize * 2] = paddedData.data[i3 + 2] * 0.003921569;
      tensor[p4 + planeSize * 2] = paddedData.data[i4 + 2] * 0.003921569;
    }

    return {
      tensor,
      paddingX: meta.paddingX,
      paddingY: meta.paddingY,
      scaleX: meta.scaleX,
      scaleY: meta.scaleY,
    };
  }

  /**
   * Postprocess YOLO output - supports multiple output formats
   */
  private postprocess(
    output: Float32Array,
    numDetections: number,
    outputShape: number[],
    imgWidth: number,
    imgHeight: number,
    paddingX: number,
    paddingY: number,
    scaleX: number,
    scaleY: number
  ): DetectedObject[] {
    const detections: DetectedObject[] = [];

    // Format 1: [batch, boxes, 6] - [x1, y1, x2, y2, conf, class]
    if (outputShape.length === 3 && outputShape[2] === 6) {
      for (let i = 0; i < numDetections; i++) {
        const idx = i * 6;
        const x1 = output[idx];
        const y1 = output[idx + 1];
        const x2 = output[idx + 2];
        const y2 = output[idx + 3];
        const confidence = output[idx + 4];
        const classId = Math.round(output[idx + 5]);

        if (confidence < this.config.confidence) continue;
        if (this.classFilter && !this.classFilter.has(classId)) continue;
        if (x2 <= x1 || y2 <= y1) continue;

        const tx1 = (x1 - paddingX) * scaleX;
        const ty1 = (y1 - paddingY) * scaleY;
        const tx2 = (x2 - paddingX) * scaleX;
        const ty2 = (y2 - paddingY) * scaleY;

        detections.push({
          bbox: {
            x1: Math.max(0, tx1),
            y1: Math.max(0, ty1),
            x2: Math.min(imgWidth, tx2),
            y2: Math.min(imgHeight, ty2),
            confidence,
          },
          classId,
          className: COCO_CLASSES[classId] || `class_${classId}`,
          confidence,
        });
      }
    }
    // Format 2: [batch, boxes, 80+] - YOLOv26 style
    // Format: [class_scores..., cx, cy, w, h] - center format with width/height
    else if (outputShape.length === 3 && outputShape[2] >= 80) {
      const numClasses = outputShape[2] - 4;
      const [inputH, inputW] = this.config.inputSize;
      
      log.debug(`Trying YOLOv26 format (center format) with ${numClasses} classes`);
      
      for (let i = 0; i < numDetections; i++) {
        const baseIdx = i * outputShape[2];
        
        // Raw bbox values - try direct interpretation first
        // YOLOv26 may output already decoded coordinates
        let x1 = output[baseIdx + numClasses];
        let y1 = output[baseIdx + numClasses + 1];
        let x2 = output[baseIdx + numClasses + 2];
        let y2 = output[baseIdx + numClasses + 3];
        
        // If values are very small (< 1), they might be logits - apply sigmoid
        if (Math.abs(x1) < 1 && Math.abs(y1) < 1) {
          // Apply sigmoid and scale
          x1 = (1 / (1 + Math.exp(-x1))) * inputW;
          y1 = (1 / (1 + Math.exp(-y1))) * inputH;
          x2 = (1 / (1 + Math.exp(-x2))) * inputW;
          y2 = (1 / (1 + Math.exp(-y2))) * inputH;
        }
        // If values are negative but large, apply sigmoid only
        else if (x1 < 0 || y1 < 0) {
          x1 = (1 / (1 + Math.exp(-x1))) * inputW;
          y1 = (1 / (1 + Math.exp(-y1))) * inputH;
          x2 = (1 / (1 + Math.exp(-x2))) * inputW;
          y2 = (1 / (1 + Math.exp(-y2))) * inputH;
        }
        // Otherwise use as-is (already decoded)
        
        // Debug first detection
        if (i === 0) {
          log.debug(`Raw bbox: [${output[baseIdx + numClasses]}, ${output[baseIdx + numClasses + 1]}, ${output[baseIdx + numClasses + 2]}, ${output[baseIdx + numClasses + 3]}]`);
          log.debug(`Decoded bbox: [${x1.toFixed(1)}, ${y1.toFixed(1)}, ${x2.toFixed(1)}, ${y2.toFixed(1)}]`);
        }
        
        // Find best class and confidence
        let bestClass = 0;
        let bestScore = -Infinity;
        
        for (let c = 0; c < numClasses; c++) {
          const score = output[baseIdx + c];
          if (score > bestScore) {
            bestScore = score;
            bestClass = c;
          }
        }
        
        // Apply sigmoid to class score
        const confidence = 1 / (1 + Math.exp(-bestScore));

        // Debug first few detections
        if (i < 5 && confidence > 0.05) {
          log.debug(`Box ${i}: [${x1.toFixed(1)}, ${y1.toFixed(1)}, ${x2.toFixed(1)}, ${y2.toFixed(1)}]`);
          log.debug(`  -> class=${bestClass} (${COCO_CLASSES[bestClass] || 'unknown'}), confidence=${(confidence * 100).toFixed(1)}%`);
        }

        if (confidence < this.config.confidence) continue;
        if (this.classFilter && !this.classFilter.has(bestClass)) continue;
        if (x2 <= x1 || y2 <= y1) continue;
        if (x1 < 0 && x2 < 0) continue;
        if (y1 < 0 && y2 < 0) continue;

        // Transform to original image space
        const tx1 = (x1 - paddingX) * scaleX;
        const ty1 = (y1 - paddingY) * scaleY;
        const tx2 = (x2 - paddingX) * scaleX;
        const ty2 = (y2 - paddingY) * scaleY;

        detections.push({
          bbox: {
            x1: Math.max(0, tx1),
            y1: Math.max(0, ty1),
            x2: Math.min(imgWidth, tx2),
            y2: Math.min(imgHeight, ty2),
            confidence,
          },
          classId: bestClass,
          className: COCO_CLASSES[bestClass] || `class_${bestClass}`,
          confidence,
        });
      }
    }

    // Debug logging
    if (detections.length > 0) {
      log.debug(`✅ Found ${detections.length} detections`);
      log.debug('First:', detections[0]);
    } else {
      log.debug(`❌ No detections above threshold ${this.config.confidence}`);
      // Log top 3 scores for debugging
      const topScores: number[] = [];
      const numClasses = outputShape.length === 3 ? outputShape[2] - 4 : 80;
      for (let i = 0; i < Math.min(3, numDetections); i++) {
        const baseIdx = i * outputShape[2];
        let bestScore = -Infinity;
        for (let c = 0; c < numClasses; c++) {
          const score = output[baseIdx + c];
          if (score > bestScore) bestScore = score;
        }
        const confidence = bestScore > 0 && bestScore <= 1 ? bestScore : 1 / (1 + Math.exp(-bestScore));
        topScores.push(confidence);
      }
      log.debug(`Top 3 confidences: ${topScores.map(s => (s * 100).toFixed(1) + '%').join(', ')}`);
    }

    // NMS
    return this.applyMultiClassNMS(detections, this.config.nmsThreshold);
  }

  /**
   * Multi-class Non-Maximum Suppression
   */
  private applyMultiClassNMS(
    detections: DetectedObject[],
    iouThreshold: number
  ): DetectedObject[] {
    if (detections.length === 0) return [];

    // Group by class
    const byClass = new Map<number, DetectedObject[]>();
    detections.forEach((det) => {
      const classDets = byClass.get(det.classId) || [];
      classDets.push(det);
      byClass.set(det.classId, classDets);
    });

    // Apply NMS per class
    const selected: DetectedObject[] = [];
    byClass.forEach((classDets) => {
      classDets.sort((a, b) => b.confidence - a.confidence);

      const used = new Set<number>();
      for (let i = 0; i < classDets.length; i++) {
        if (used.has(i)) continue;

        selected.push(classDets[i]);
        used.add(i);

        for (let j = i + 1; j < classDets.length; j++) {
          if (used.has(j)) continue;

          const iou = this.calculateIoU(classDets[i].bbox, classDets[j].bbox);
          if (iou > iouThreshold) {
            used.add(j);
          }
        }
      }
    });

    return selected;
  }

  /**
   * Calculate IoU between two boxes
   */
  private calculateIoU(
    box1: { x1: number; y1: number; x2: number; y2: number },
    box2: { x1: number; y1: number; x2: number; y2: number }
  ): number {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);

    if (x2 <= x1 || y2 <= y1) return 0;

    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    const union = area1 + area2 - intersection;

    return intersection / union;
  }

  /**
   * Calculate detection statistics
   */
  private calculateStats(
    detections: DetectedObject[],
    inferenceTime: number
  ): DetectionStats {
    const classCounts: Record<string, number> = {};

    detections.forEach((det) => {
      classCounts[det.className] = (classCounts[det.className] || 0) + 1;
    });

    return {
      totalCount: detections.length,
      classCounts,
      inferenceTime: Math.round(inferenceTime),
    };
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    if (this.session) {
      this.session.release();
      this.session = null;
    }
    if (this.mediaPipeDetector) {
      this.mediaPipeDetector.dispose();
      this.mediaPipeDetector = null;
    }
    this.initialized = false;
  }
}
