/**
 * PoseDetector - Unified API for person detection and pose estimation
 * Combines YOLO12 detector with RTMW pose model in a single optimized interface
 *
 * @example
 * ```typescript
 * // Initialize with default models (from HuggingFace)
 * const detector = new PoseDetector();
 * await detector.init();
 *
 * // Or with custom models
 * const detector = new PoseDetector({
 *   detModel: 'models/yolov12n.onnx',
 *   poseModel: 'models/rtmlib/end2end.onnx',
 * });
 * await detector.init();
 *
 * // From canvas
 * const results = await detector.detectFromCanvas(canvas);
 *
 * // From video element
 * const results = await detector.detectFromVideo(videoElement);
 *
 * // From raw image data
 * const results = await detector.detect(imageData, width, height);
 * ```
 */

import * as ort from 'onnxruntime-web/all';
import { BBox, Detection, type WebNNProviderOptions } from '../types/index';
import { getCachedModel, isModelCached } from '../core/modelCache';
import { MediaPipeObjectDetector } from './mediaPipeObjectDetector';
import { initOnnxRuntimeWeb } from '../core/onnxRuntime';

// Configure ONNX Runtime Web (only in browser environment)
initOnnxRuntimeWeb();

/**
 * Backend type for PoseDetector
 */
export type PoseDetectorBackend = 'yolo-rtmpose' | 'mediapipe-rtmpose';

/**
 * Configuration options for PoseDetector
 */
export interface PoseDetectorConfig {
  /** Path to YOLO12 detection model (optional - uses default from HuggingFace if not specified) */
  detModel?: string;
  /** Path to RTMW pose estimation model (optional - uses default from HuggingFace if not specified) */
  poseModel?: string;
  /** Detection input size (default: [416, 416]) */
  detInputSize?: [number, number];
  /** Pose input size (default: [384, 288]) */
  poseInputSize?: [number, number];
  /** Detection confidence threshold (default: 0.5) */
  detConfidence?: number;
  /** NMS IoU threshold (default: 0.45) */
  nmsThreshold?: number;
  /** Pose keypoint confidence threshold (default: 0.3) */
  poseConfidence?: number;
  /** Execution backend (default: 'webgl') */
  backend?: 'wasm' | 'webgl' | 'webgpu' | 'webnn';
  /** WebNN provider options (only used when backend is 'webnn') */
  webnnOptions?: import('../types/index').WebNNProviderOptionsOrUndefined;
  /** Device type for WebNN/WebGPU (default: 'gpu' for high performance) */
  deviceType?: 'cpu' | 'gpu' | 'npu';
  /** Power preference for WebNN/WebGPU (default: 'high-performance') */
  powerPreference?: 'default' | 'low-power' | 'high-performance';
  /** Enable model caching (default: true) */
  cache?: boolean;
  /** Backend type: 'yolo-rtmpose' for YOLO detection, 'mediapipe-rtmpose' for MediaPipe detection + RTMPose (default: 'yolo-rtmpose') */
  detectorType?: PoseDetectorBackend;
  /** MediaPipe model path for person detection (only used when detectorType is 'mediapipe-rtmpose') */
  mediaPipeModelPath?: string;
  /** MediaPipe score threshold (only used when detectorType is 'mediapipe-rtmpose') */
  mediaPipeScoreThreshold?: number;
  /** MediaPipe max results (only used when detectorType is 'mediapipe-rtmpose') */
  mediaPipeMaxResults?: number;
}

/**
 * Detected person with bounding box and keypoints
 */
export interface Person {
  /** Bounding box coordinates */
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
  };
  /** 17 COCO keypoints coordinates */
  keypoints: Keypoint[];
  /** Keypoint scores (0-1) */
  scores: number[];
}

/**
 * Single keypoint with coordinates and visibility
 */
export interface Keypoint {
  x: number;
  y: number;
  score: number;
  visible: boolean;
  name: string;
}

/**
 * Detection statistics
 */
export interface PoseStats {
  /** Number of detected people */
  personCount: number;
  /** Detection inference time (ms) */
  detTime: number;
  /** Pose estimation time (ms) */
  poseTime: number;
  /** Total processing time (ms) */
  totalTime: number;
}

/**
 * COCO17 keypoint names
 */
const KEYPOINT_NAMES = [
  'nose',
  'left_eye',
  'right_eye',
  'left_ear',
  'right_ear',
  'left_shoulder',
  'right_shoulder',
  'left_elbow',
  'right_elbow',
  'left_wrist',
  'right_wrist',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle',
];

/**
 * Default configuration
 */
const DEFAULT_CONFIG: Omit<Required<PoseDetectorConfig>, 'webnnOptions'> & {
  webnnOptions?: import('../types/index').WebNNProviderOptionsOrUndefined;
} = {
  detModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  poseModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/rtmpose/end2end.onnx',
  detInputSize: [416, 416],
  poseInputSize: [384, 288],
  detConfidence: 0.5,
  nmsThreshold: 0.45,
  poseConfidence: 0.3,
  backend: 'webgl',
  webnnOptions: undefined as import('../types/index').WebNNProviderOptionsOrUndefined,
  deviceType: 'gpu',
  powerPreference: 'high-performance',
  cache: true,
  detectorType: 'yolo-rtmpose',
  mediaPipeModelPath: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite',
  mediaPipeScoreThreshold: 0.5,
  mediaPipeMaxResults: -1,
};

export class PoseDetector {
  private config: Omit<Required<PoseDetectorConfig>, 'webnnOptions'> & {
    webnnOptions?: import('../types/index').WebNNProviderOptionsOrUndefined;
  };
  private detSession: ort.InferenceSession | null = null;
  private poseSession: ort.InferenceSession | null = null;
  private mediaPipeObjectDetector: MediaPipeObjectDetector | null = null;
  private initialized = false;

  // Pre-allocated buffers for maximum performance
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private poseCanvas: HTMLCanvasElement | null = null;
  private poseCtx: CanvasRenderingContext2D | null = null;
  private poseTensorBuffer: Float32Array | null = null;
  private detInputSize: [number, number] = [416, 416];
  private poseInputSize: [number, number] = [384, 288];

  constructor(config: PoseDetectorConfig) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Initialize both detection and pose models with pre-allocated resources
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    try {
      // Initialize based on detector type
      if (this.config.detectorType === 'mediapipe-rtmpose') {
        await this.initMediaPipe();
      } else {
        await this.initYoloRtmpose();
      }
    } catch (error) {
      console.error('[PoseDetector] ❌ Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Initialize YOLO + RTMPose models (original implementation)
   */
  private async initYoloRtmpose(): Promise<void> {
    // Load detection model
    console.log(`[PoseDetector] Loading detection model from: ${this.config.detModel}`);
    let detBuffer: ArrayBuffer;

    if (this.config.cache) {
      const detCached = await isModelCached(this.config.detModel);
      console.log(`[PoseDetector] Det model cache ${detCached ? 'hit' : 'miss'}`);
      detBuffer = await getCachedModel(this.config.detModel);
    } else {
      const detResponse = await fetch(this.config.detModel);
      if (!detResponse.ok) {
        throw new Error(`Failed to fetch det model: HTTP ${detResponse.status}`);
      }
      detBuffer = await detResponse.arrayBuffer();
    }

    // Build execution providers with WebNN options
    const detExecProviders: any[] = [];
    
    if (this.config.backend === 'webnn') {
      const webnnOptions = {
        name: 'webnn' as const,
        deviceType: this.config.deviceType || 'gpu',
        powerPreference: this.config.powerPreference || 'high-performance',
      };
      detExecProviders.push(webnnOptions);
      console.log(`[PoseDetector] ✅ Detection model using WebNN: deviceType=${webnnOptions.deviceType}, powerPreference=${webnnOptions.powerPreference}`);
    } else if (this.config.backend === 'webgpu') {
      // Check if WebGPU is available
      if (typeof navigator !== 'undefined' && (navigator as any).gpu) {
        detExecProviders.push('webgpu');
        console.log(`[PoseDetector] ✅ Detection model using WebGPU`);
      } else {
        console.warn(`[PoseDetector] ⚠️ WebGPU not available, falling back to WebGL`);
        detExecProviders.push('webgl');
      }
    } else {
      detExecProviders.push(this.config.backend);
      console.log(`[PoseDetector] ✅ Detection model using backend: ${this.config.backend}`);
    }

    console.log(`[PoseDetector] Detection execution providers: ${JSON.stringify(detExecProviders)}`);

    this.detSession = await ort.InferenceSession.create(detBuffer, {
      executionProviders: detExecProviders,
      graphOptimizationLevel: 'all',
    });
    console.log(`[PoseDetector] ✅ Detection session created successfully`);
    console.log(`[PoseDetector] Detection model loaded, size: ${(detBuffer.byteLength / 1024 / 1024).toFixed(2)} MB`);

    // Load pose model
    console.log(`[PoseDetector] Loading pose model from: ${this.config.poseModel}`);
    let poseBuffer: ArrayBuffer;

    if (this.config.cache) {
      const poseCached = await isModelCached(this.config.poseModel);
      console.log(`[PoseDetector] Pose model cache ${poseCached ? 'hit' : 'miss'}`);
      poseBuffer = await getCachedModel(this.config.poseModel);
    } else {
      const poseResponse = await fetch(this.config.poseModel);
      if (!poseResponse.ok) {
        throw new Error(`Failed to fetch pose model: HTTP ${poseResponse.status}`);
      }
      poseBuffer = await poseResponse.arrayBuffer();
    }

    const poseExecProviders: any[] = [];
    
    if (this.config.backend === 'webnn') {
      const webnnOptions = {
        name: 'webnn' as const,
        deviceType: this.config.deviceType || 'gpu',
        powerPreference: this.config.powerPreference || 'high-performance',
      };
      poseExecProviders.push(webnnOptions);
      console.log(`[PoseDetector] ✅ Pose model using WebNN: deviceType=${webnnOptions.deviceType}, powerPreference=${webnnOptions.powerPreference}`);
    } else if (this.config.backend === 'webgpu') {
      // Check if WebGPU is available
      if (typeof navigator !== 'undefined' && (navigator as any).gpu) {
        poseExecProviders.push('webgpu');
        console.log(`[PoseDetector] ✅ Pose model using WebGPU`);
      } else {
        console.warn(`[PoseDetector] ⚠️ WebGPU not available, falling back to WebGL`);
        poseExecProviders.push('webgl');
      }
    } else {
      poseExecProviders.push(this.config.backend);
      console.log(`[PoseDetector] ✅ Pose model using backend: ${this.config.backend}`);
    }

    console.log(`[PoseDetector] Pose execution providers: ${JSON.stringify(poseExecProviders)}`);

    this.poseSession = await ort.InferenceSession.create(poseBuffer, {
      executionProviders: poseExecProviders,
      graphOptimizationLevel: 'all',
    });
    console.log(`[PoseDetector] ✅ Pose session created successfully`);
    console.log(`[PoseDetector] Pose model loaded, size: ${(poseBuffer.byteLength / 1024 / 1024).toFixed(2)} MB`);

    // Pre-allocate all resources
    const [detW, detH] = this.config.detInputSize;
    this.detInputSize = [detW, detH];

    const [poseW, poseH] = this.config.poseInputSize;
    this.poseInputSize = [poseW, poseH];

    // Main canvas for detection
    this.canvas = document.createElement('canvas');
    this.canvas.width = detW;
    this.canvas.height = detH;
    this.ctx = this.canvas.getContext('2d', {
      willReadFrequently: true,
      alpha: false
    })!;

    // Pose crop canvas (reused for each person)
    this.poseCanvas = document.createElement('canvas');
    this.poseCanvas.width = poseW;
    this.poseCanvas.height = poseH;
    this.poseCtx = this.poseCanvas.getContext('2d', {
      willReadFrequently: true,
      alpha: false
    })!;

    // Pre-allocate pose tensor buffer
    this.poseTensorBuffer = new Float32Array(3 * poseW * poseH);

    this.initialized = true;
    console.log(`[PoseDetector] ✅ YOLO+RTMPose Initialized (det:${detW}x${detH}, pose:${poseW}x${poseH})`);
  }

  /**
   * Initialize MediaPipe Object Detector + RTMPose
   */
  private async initMediaPipe(): Promise<void> {
    // Initialize MediaPipe Object Detector for person detection
    this.mediaPipeObjectDetector = new MediaPipeObjectDetector({
      modelPath: this.config.mediaPipeModelPath,
      scoreThreshold: this.config.mediaPipeScoreThreshold,
      maxResults: this.config.mediaPipeMaxResults,
      categoryAllowlist: ['person'],
      cache: this.config.cache,
    });

    await this.mediaPipeObjectDetector.init();

    // Initialize RTMPose model
    let poseBuffer: ArrayBuffer;
    if (this.config.cache) {
      poseBuffer = await getCachedModel(this.config.poseModel);
    } else {
      const response = await fetch(this.config.poseModel);
      if (!response.ok) throw new Error(`Failed to fetch pose model: HTTP ${response.status}`);
      poseBuffer = await response.arrayBuffer();
    }

    this.poseSession = await ort.InferenceSession.create(poseBuffer, {
      executionProviders: [this.config.backend, 'wasm'],
      graphOptimizationLevel: 'all',
    });

    const [poseW, poseH] = this.config.poseInputSize;
    this.poseInputSize = [poseW, poseH];

    this.poseCanvas = document.createElement('canvas');
    this.poseCanvas.width = poseW;
    this.poseCanvas.height = poseH;
    this.poseCtx = this.poseCanvas.getContext('2d', { willReadFrequently: true, alpha: false })!;
    this.poseTensorBuffer = new Float32Array(3 * poseW * poseH);

    this.initialized = true;
  }

  /**
   * Detect poses from HTMLCanvasElement
   * @param canvas - Canvas element containing the image
   * @returns Array of detected people with keypoints
   */
  async detectFromCanvas(canvas: HTMLCanvasElement): Promise<Person[]> {
    if (!this.initialized) await this.init();

    // Use MediaPipe Object Detector if selected
    if (this.config.detectorType === 'mediapipe-rtmpose' && this.mediaPipeObjectDetector) {
      const mpDetections = await this.mediaPipeObjectDetector.detectFromCanvas(canvas);
      return this.detectPosesFromDetections(canvas, mpDetections.map(d => ({
        x1: d.bbox.x1,
        y1: d.bbox.y1,
        x2: d.bbox.x1 + d.bbox.width,
        y2: d.bbox.y1 + d.bbox.height,
        confidence: d.score,
      })));
    }

    // Otherwise use YOLO
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D context from canvas');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return this.detect(new Uint8Array(imageData.data.buffer), canvas.width, canvas.height);
  }

  /**
   * Detect poses from HTMLVideoElement
   * @param video - Video element to capture frame from
   * @param targetCanvas - Optional canvas for frame extraction (creates one if not provided)
   * @returns Array of detected people with keypoints
   */
  async detectFromVideo(
    video: HTMLVideoElement,
    targetCanvas?: HTMLCanvasElement
  ): Promise<Person[]> {
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
   * Detect poses from HTMLImageElement
   * @param image - Image element to process
   * @param targetCanvas - Optional canvas for image extraction (creates one if not provided)
   * @returns Array of detected people with keypoints
   */
  async detectFromImage(
    image: HTMLImageElement,
    targetCanvas?: HTMLCanvasElement
  ): Promise<Person[]> {
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
   * Detect poses from ImageBitmap (efficient for blob/file uploads)
   * @param bitmap - ImageBitmap to process
   * @param targetCanvas - Optional canvas for bitmap extraction (creates one if not provided)
   * @returns Array of detected people with keypoints
   */
  async detectFromBitmap(
    bitmap: ImageBitmap,
    targetCanvas?: HTMLCanvasElement
  ): Promise<Person[]> {
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
   * Detect poses from File (for file input uploads)
   * @param file - File object from input element
   * @param targetCanvas - Optional canvas for image extraction (creates one if not provided)
   * @returns Array of detected people with keypoints
   */
  async detectFromFile(
    file: File,
    targetCanvas?: HTMLCanvasElement
  ): Promise<Person[]> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = async () => {
        try {
          const results = await this.detectFromImage(img, targetCanvas);
          resolve(results);
        } catch (error) {
          reject(error);
        }
      };
      img.onerror = () => reject(new Error('Failed to load image from file'));
      img.src = URL.createObjectURL(file);
    });
  }

  /**
   * Detect poses from Blob (for camera capture or downloads)
   * @param blob - Blob object to process
   * @param targetCanvas - Optional canvas for image extraction (creates one if not provided)
   * @returns Array of detected people with keypoints
   */
  async detectFromBlob(
    blob: Blob,
    targetCanvas?: HTMLCanvasElement
  ): Promise<Person[]> {
    const bitmap = await createImageBitmap(blob);
    const results = await this.detectFromBitmap(bitmap, targetCanvas);
    bitmap.close();
    return results;
  }

  /**
   * Detect people and estimate poses in a single call
   * @param imageData - Image data (Uint8Array RGB/RGBA)
   * @param width - Image width
   * @param height - Image height
   * @returns Array of detected people with keypoints
   */
  async detect(
    imageData: Uint8Array,
    width: number,
    height: number
  ): Promise<Person[]> {
    if (!this.initialized) {
      await this.init();
    }

    const startTime = performance.now();

    // Step 1: Detect people
    const detStart = performance.now();
    const bboxes = await this.detectPeople(imageData, width, height);
    const detTime = performance.now() - detStart;

    // Step 2: Estimate poses for each person
    const poseStart = performance.now();
    const people: Person[] = [];

    for (const bbox of bboxes) {
      const keypoints = await this.estimatePose(imageData, width, height, bbox);
      people.push({
        bbox: {
          x1: bbox.x1,
          y1: bbox.y1,
          x2: bbox.x2,
          y2: bbox.y2,
          confidence: bbox.confidence,
        },
        keypoints,
        scores: keypoints.map(k => k.score),
      });
    }

    const poseTime = performance.now() - poseStart;
    const totalTime = performance.now() - startTime;

    // Attach stats (for debugging)
    (people as any).stats = {
      personCount: people.length,
      detTime: Math.round(detTime),
      poseTime: Math.round(poseTime),
      totalTime: Math.round(totalTime),
    } as PoseStats;

    return people;
  }

  /**
   * Get detection and pose statistics from last call
   */
  getStats(): PoseStats | null {
    return null;
  }

  /**
   * Estimate poses from pre-detected bounding boxes
   */
  private async detectPosesFromDetections(
    canvas: HTMLCanvasElement,
    bboxes: Array<{ x1: number; y1: number; x2: number; y2: number; confidence: number }>
  ): Promise<Person[]> {
    const imageData = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
    const poseStart = performance.now();
    const people: Person[] = [];

    for (const bbox of bboxes) {
      const keypoints = await this.estimatePose(
        new Uint8Array(imageData.data.buffer),
        canvas.width,
        canvas.height,
        bbox
      );
      people.push({
        bbox,
        keypoints,
        scores: keypoints.map(k => k.score),
      });
    }

    const poseTime = performance.now() - poseStart;

    (people as any).stats = {
      personCount: people.length,
      detTime: 0,
      poseTime: Math.round(poseTime),
      totalTime: Math.round(poseTime),
    } as PoseStats;

    return people;
  }

  /**
   * Detect people using YOLO12
   */
  private async detectPeople(
    imageData: Uint8Array,
    width: number,
    height: number
  ): Promise<Array<{ x1: number; y1: number; x2: number; y2: number; confidence: number }>> {
    const [inputH, inputW] = this.config.detInputSize;

    // Preprocess
    const { tensor, paddingX, paddingY, scaleX, scaleY } = this.preprocessYOLO(
      imageData,
      width,
      height,
      [inputW, inputH]
    );

    // Inference - use dynamic input name
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, inputH, inputW]);
    const inputName = this.detSession!.inputNames[0];  // Dynamic: 'images' or 'pixel_values'
    
    const feeds: Record<string, ort.Tensor> = {};
    feeds[inputName] = inputTensor;
    
    const results = await this.detSession!.run(feeds);
    const output = results[this.detSession!.outputNames[0]];

    // Postprocess
    return this.postprocessYOLO(
      output.data as Float32Array,
      output.dims[1],
      width,
      height,
      paddingX,
      paddingY,
      scaleX,
      scaleY
    );
  }

  /**
   * Estimate pose for a single person
   */
  private async estimatePose(
    imageData: Uint8Array,
    imgWidth: number,
    imgHeight: number,
    bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number }
  ): Promise<Keypoint[]> {
    const [inputH, inputW] = this.config.poseInputSize;

    // Preprocess
    const { tensor, center, scale } = this.preprocessPose(
      imageData,
      imgWidth,
      imgHeight,
      bbox,
      [inputW, inputH]
    );

    // Inference
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, inputH, inputW] as number[]);
    const results = await this.poseSession!.run({ input: inputTensor });

    // Postprocess
    return this.postprocessPose(
      results.simcc_x.data as Float32Array,
      results.simcc_y.data as Float32Array,
      results.simcc_x.dims as number[],
      results.simcc_y.dims as number[],
      center,
      scale
    );
  }

  /**
   * YOLO preprocessing with letterbox
   */
  private preprocessYOLO(
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

    // Reuse canvas
    if (!this.canvas || !this.ctx) {
      this.canvas = document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })!;
    }

    this.canvas.width = inputW;
    this.canvas.height = inputH;
    const ctx = this.ctx;

    // Black background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, inputW, inputH);

    // Calculate letterbox
    const aspectRatio = imgWidth / imgHeight;
    const targetAspectRatio = inputW / inputH;

    let drawWidth: number, drawHeight: number, offsetX: number, offsetY: number;

    if (aspectRatio > targetAspectRatio) {
      drawWidth = inputW;
      drawHeight = Math.floor(inputW / aspectRatio);
      offsetX = 0;
      offsetY = Math.floor((inputH - drawHeight) / 2);
    } else {
      drawHeight = inputH;
      drawWidth = Math.floor(inputH * aspectRatio);
      offsetX = Math.floor((inputW - drawWidth) / 2);
      offsetY = 0;
    }

    // Create source canvas
    const srcCanvas = document.createElement('canvas');
    const srcCtx = srcCanvas.getContext('2d')!;
    srcCanvas.width = imgWidth;
    srcCanvas.height = imgHeight;

    const srcImageData = srcCtx.createImageData(imgWidth, imgHeight);
    srcImageData.data.set(imageData);
    srcCtx.putImageData(srcImageData, 0, 0);

    // Draw
    ctx.drawImage(srcCanvas, 0, 0, imgWidth, imgHeight, offsetX, offsetY, drawWidth, drawHeight);

    const paddedData = ctx.getImageData(0, 0, inputW, inputH);

    // Normalize to [0, 1] and convert to CHW
    const tensor = new Float32Array(inputW * inputH * 3);
    for (let i = 0; i < paddedData.data.length; i += 4) {
      const pixelIdx = i / 4;
      tensor[pixelIdx] = paddedData.data[i] / 255;
      tensor[pixelIdx + inputW * inputH] = paddedData.data[i + 1] / 255;
      tensor[pixelIdx + 2 * inputW * inputH] = paddedData.data[i + 2] / 255;
    }

    const scaleX = imgWidth / drawWidth;
    const scaleY = imgHeight / drawHeight;

    return {
      tensor,
      paddingX: offsetX,
      paddingY: offsetY,
      scaleX,
      scaleY,
    };
  }

  /**
   * YOLO postprocessing with NMS
   */
  private postprocessYOLO(
    output: Float32Array,
    numDetections: number,
    imgWidth: number,
    imgHeight: number,
    paddingX: number,
    paddingY: number,
    scaleX: number,
    scaleY: number
  ): Array<{ x1: number; y1: number; x2: number; y2: number; confidence: number }> {
    const detections: Array<{ x1: number; y1: number; x2: number; y2: number; confidence: number }> = [];

    for (let i = 0; i < numDetections; i++) {
      const idx = i * 6;
      const x1 = output[idx];
      const y1 = output[idx + 1];
      const x2 = output[idx + 2];
      const y2 = output[idx + 3];
      const confidence = output[idx + 4];
      const classId = Math.round(output[idx + 5]);

      if (confidence < this.config.detConfidence || classId !== 0) continue;

      // Transform coordinates
      const tx1 = (x1 - paddingX) * scaleX;
      const ty1 = (y1 - paddingY) * scaleY;
      const tx2 = (x2 - paddingX) * scaleX;
      const ty2 = (y2 - paddingY) * scaleY;

      detections.push({
        x1: Math.max(0, tx1),
        y1: Math.max(0, ty1),
        x2: Math.min(imgWidth, tx2),
        y2: Math.min(imgHeight, ty2),
        confidence,
      });
    }

    // NMS
    return this.applyNMS(detections, this.config.nmsThreshold);
  }

  /**
   * Pose preprocessing with affine crop
   */
  private preprocessPose(
    imageData: Uint8Array,
    imgWidth: number,
    imgHeight: number,
    bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number },
    inputSize: [number, number]
  ): { tensor: Float32Array; center: [number, number]; scale: [number, number] } {
    const [inputW, inputH] = inputSize;

    const bboxWidth = bbox.x2 - bbox.x1;
    const bboxHeight = bbox.y2 - bbox.y1;

    const center: [number, number] = [
      bbox.x1 + bboxWidth / 2,
      bbox.y1 + bboxHeight / 2,
    ];

    // Aspect ratio preservation
    const bboxAspectRatio = bboxWidth / bboxHeight;
    const modelAspectRatio = inputW / inputH;

    let scaleW: number, scaleH: number;
    if (bboxAspectRatio > modelAspectRatio) {
      scaleW = bboxWidth * 1.25;
      scaleH = scaleW / modelAspectRatio;
    } else {
      scaleH = bboxHeight * 1.25;
      scaleW = scaleH * modelAspectRatio;
    }

    const scale: [number, number] = [scaleW, scaleH];

    // Reuse pre-allocated pose canvas
    if (!this.poseCanvas || !this.poseCtx) {
      this.poseCanvas = document.createElement('canvas');
      this.poseCanvas.width = inputW;
      this.poseCanvas.height = inputH;
      this.poseCtx = this.poseCanvas.getContext('2d', {
        willReadFrequently: true,
        alpha: false
      })!;
      this.poseTensorBuffer = new Float32Array(3 * inputW * inputH);
    }

    const ctx = this.poseCtx;

    // Fast clear
    ctx.clearRect(0, 0, inputW, inputH);

    // Create source
    const srcCanvas = document.createElement('canvas');
    const srcCtx = srcCanvas.getContext('2d')!;
    srcCanvas.width = imgWidth;
    srcCanvas.height = imgHeight;

    const srcImageData = srcCtx.createImageData(imgWidth, imgHeight);
    srcImageData.data.set(imageData);
    srcCtx.putImageData(srcImageData, 0, 0);

    // Crop and scale
    const srcX = center[0] - scaleW / 2;
    const srcY = center[1] - scaleH / 2;
    ctx.drawImage(srcCanvas, srcX, srcY, scaleW, scaleH, 0, 0, inputW, inputH);

    const croppedData = ctx.getImageData(0, 0, inputW, inputH);

    // Optimized normalization with precomputed constants
    const tensor = this.poseTensorBuffer!;
    const len = croppedData.data.length;
    const planeSize = inputW * inputH;
    
    // Precompute normalization constants
    const mean0 = 123.675, mean1 = 116.28, mean2 = 103.53;
    const stdInv0 = 1 / 58.395, stdInv1 = 1 / 57.12, stdInv2 = 1 / 57.375;
    
    // Unrolled loop (4 pixels at once)
    for (let i = 0; i < len; i += 16) {
      const p1 = i / 4, p2 = p1 + 1, p3 = p1 + 2, p4 = p1 + 3;
      
      // R channel
      tensor[p1] = (croppedData.data[i] - mean0) * stdInv0;
      tensor[p2] = (croppedData.data[i + 4] - mean0) * stdInv0;
      tensor[p3] = (croppedData.data[i + 8] - mean0) * stdInv0;
      tensor[p4] = (croppedData.data[i + 12] - mean0) * stdInv0;
      
      // G channel
      tensor[p1 + planeSize] = (croppedData.data[i + 1] - mean1) * stdInv1;
      tensor[p2 + planeSize] = (croppedData.data[i + 5] - mean1) * stdInv1;
      tensor[p3 + planeSize] = (croppedData.data[i + 9] - mean1) * stdInv1;
      tensor[p4 + planeSize] = (croppedData.data[i + 13] - mean1) * stdInv1;
      
      // B channel
      tensor[p1 + planeSize * 2] = (croppedData.data[i + 2] - mean2) * stdInv2;
      tensor[p2 + planeSize * 2] = (croppedData.data[i + 6] - mean2) * stdInv2;
      tensor[p3 + planeSize * 2] = (croppedData.data[i + 10] - mean2) * stdInv2;
      tensor[p4 + planeSize * 2] = (croppedData.data[i + 14] - mean2) * stdInv2;
    }

    return { tensor, center, scale };
  }

  /**
   * Pose postprocessing with SimCC decoding
   */
  private postprocessPose(
    simccX: Float32Array,
    simccY: Float32Array,
    shapeX: number[],
    shapeY: number[],
    center: [number, number],
    scale: [number, number]
  ): Keypoint[] {
    const numKeypoints = shapeX[1];
    const wx = shapeX[2];
    const wy = shapeY[2];

    const keypoints: Keypoint[] = [];

    for (let k = 0; k < numKeypoints; k++) {
      // Argmax X
      let maxX = -Infinity;
      let argmaxX = 0;
      for (let i = 0; i < wx; i++) {
        const val = simccX[k * wx + i];
        if (val > maxX) {
          maxX = val;
          argmaxX = i;
        }
      }

      // Argmax Y
      let maxY = -Infinity;
      let argmaxY = 0;
      for (let i = 0; i < wy; i++) {
        const val = simccY[k * wy + i];
        if (val > maxY) {
          maxY = val;
          argmaxY = i;
        }
      }

      const score = 0.5 * (maxX + maxY);
      const visible = score > this.config.poseConfidence;

      // Transform to original coordinates
      const normX = argmaxX / wx;
      const normY = argmaxY / wy;

      const x = (normX - 0.5) * scale[0] + center[0];
      const y = (normY - 0.5) * scale[1] + center[1];

      keypoints.push({
        x,
        y,
        score,
        visible,
        name: KEYPOINT_NAMES[k] || `keypoint_${k}`,
      });
    }

    return keypoints;
  }

  /**
   * Non-Maximum Suppression
   */
  private applyNMS(
    detections: Array<{ x1: number; y1: number; x2: number; y2: number; confidence: number }>,
    iouThreshold: number
  ): typeof detections {
    if (detections.length === 0) return [];

    detections.sort((a, b) => b.confidence - a.confidence);

    const selected: typeof detections = [];
    const used = new Set<number>();

    for (let i = 0; i < detections.length; i++) {
      if (used.has(i)) continue;

      selected.push(detections[i]);
      used.add(i);

      for (let j = i + 1; j < detections.length; j++) {
        if (used.has(j)) continue;

        const iou = this.calculateIoU(detections[i], detections[j]);
        if (iou > iouThreshold) {
          used.add(j);
        }
      }
    }

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
   * Dispose resources
   */
  dispose(): void {
    if (this.detSession) {
      this.detSession.release();
      this.detSession = null;
    }
    if (this.poseSession) {
      this.poseSession.release();
      this.poseSession = null;
    }
    if (this.mediaPipeObjectDetector) {
      this.mediaPipeObjectDetector.dispose();
      this.mediaPipeObjectDetector = null;
    }
    this.initialized = false;
  }
}
