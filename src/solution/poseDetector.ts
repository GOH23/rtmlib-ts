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

import * as ort from 'onnxruntime-web';
import { BBox, Detection } from '../types/index';
import { getCachedModel, isModelCached } from '../core/modelCache';

// Configure ONNX Runtime Web
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';
ort.env.wasm.simd = true;
ort.env.wasm.proxy = false;

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
  /** Execution backend (default: 'wasm') */
  backend?: 'wasm' | 'webgpu';
  /** Enable model caching (default: true) */
  cache?: boolean;
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
const DEFAULT_CONFIG: Required<PoseDetectorConfig> = {
  detModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  poseModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/rtmpose/end2end.onnx',
  detInputSize: [416, 416],  // Faster detection
  poseInputSize: [384, 288],  // Required by model
  detConfidence: 0.5,
  nmsThreshold: 0.45,
  poseConfidence: 0.3,
  backend: 'webgpu',  // Default to WebGPU for better performance
  cache: true,
};

export class PoseDetector {
  private config: Required<PoseDetectorConfig>;
  private detSession: ort.InferenceSession | null = null;
  private poseSession: ort.InferenceSession | null = null;
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
      
      this.detSession = await ort.InferenceSession.create(detBuffer, {
        executionProviders: [this.config.backend],
        graphOptimizationLevel: 'all',
      });
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
      
      this.poseSession = await ort.InferenceSession.create(poseBuffer, {
        executionProviders: [this.config.backend],
        graphOptimizationLevel: 'all',
      });
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
      console.log(`[PoseDetector] ✅ Initialized (det:${detW}x${detH}, pose:${poseW}x${poseH})`);
    } catch (error) {
      console.error('[PoseDetector] ❌ Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Detect poses from HTMLCanvasElement
   * @param canvas - Canvas element containing the image
   * @returns Array of detected people with keypoints
   */
  async detectFromCanvas(canvas: HTMLCanvasElement): Promise<Person[]> {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context from canvas');
    }

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
    return null; // Stats attached to results
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
    this.initialized = false;
  }
}
