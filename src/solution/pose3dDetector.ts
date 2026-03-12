/**
 * Pose3DDetector - 3D Pose Estimation API
 * Combines YOLOX detector with RTMW3D 3D pose model
 *
 * @example
 * ```typescript
 * // Initialize with default models
 * const detector = new Pose3DDetector();
 * await detector.init();
 *
 * // From canvas
 * const result = await detector.detectFromCanvas(canvas);
 * console.log(result.keypoints[0][0]); // [x, y, z] - 3D coordinates
 *
 * // With custom models
 * const detector2 = new Pose3DDetector({
 *   detModel: 'path/to/yolox.onnx',
 *   poseModel: 'path/to/rtmw3d.onnx',
 * });
 * ```
 */

import * as ort from 'onnxruntime-web';
import { getCachedModel, isModelCached } from '../core/modelCache';
import { Wholebody3DResult } from './wholebody3d';

// Configure ONNX Runtime Web
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';
ort.env.wasm.simd = true;
ort.env.wasm.proxy = false;

/**
 * Configuration options for Pose3DDetector
 */
export interface Pose3DDetectorConfig {
  /** Path to YOLOX detection model (optional - uses default if not specified) */
  detModel?: string;
  /** Path to RTMW3D 3D pose estimation model (optional - uses default if not specified) */
  poseModel?: string;
  /** Detection input size (default: [640, 640]) */
  detInputSize?: [number, number];
  /** Pose input size (default: [384, 288]) */
  poseInputSize?: [number, number];
  /** Detection confidence threshold (default: 0.45) */
  detConfidence?: number;
  /** NMS IoU threshold (default: 0.7) */
  nmsThreshold?: number;
  /** Pose keypoint confidence threshold (default: 0.3) */
  poseConfidence?: number;
  /** Execution backend (default: 'webgpu') */
  backend?: 'wasm' | 'webgpu';
  /** Enable model caching (default: true) */
  cache?: boolean;
  /** Z-axis range in meters (default: 2.1744869) */
  zRange?: number;
}

/**
 * 3D Person result with 3D keypoints
 */
export interface Person3D {
  /** Bounding box coordinates */
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
  };
  /** 17 3D keypoints [x, y, z] in meters */
  keypoints: number[][];
  /** Keypoint scores (0-1) */
  scores: number[];
  /** 2D projection of keypoints */
  keypoints2d: number[][];
  /** Normalized SimCC coordinates */
  keypointsSimcc: number[][];
}

/**
 * Detection statistics
 */
export interface Pose3DStats {
  /** Number of detected people */
  personCount: number;
  /** Detection inference time (ms) */
  detTime: number;
  /** 3D Pose estimation time (ms) */
  poseTime: number;
  /** Total processing time (ms) */
  totalTime: number;
}

/**
 * COCO17 keypoint names
 */
const KEYPOINT_NAMES_3D = [
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
 * Default configuration - uses HuggingFace models
 */
const DEFAULT_CONFIG: Required<Pose3DDetectorConfig> = {
  detModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  poseModel: 'https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx',
  detInputSize: [640, 640],
  poseInputSize: [288, 384],  // [width=288, height=384] - creates tensor [1,3,384,288]
  detConfidence: 0.45,
  nmsThreshold: 0.7,
  poseConfidence: 0.3,
  backend: 'webgpu',  // Default to WebGPU for better performance
  cache: true,
  zRange: 2.1744869,
};

export class Pose3DDetector {
  private config: Required<Pose3DDetectorConfig>;
  private detSession: ort.InferenceSession | null = null;
  private poseSession: ort.InferenceSession | null = null;
  private initialized = false;
  private outputNamesLogged = false;

  // Pre-allocated buffers for better performance
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private poseCanvas: HTMLCanvasElement | null = null;
  private poseCtx: CanvasRenderingContext2D | null = null;
  private poseTensorBuffer: Float32Array | null = null;
  private detInputSize: [number, number] = [640, 640];
  private poseInputSize: [number, number] = [288, 384];  // [width=288, height=384]
  
  // Pre-allocated source canvas for pose cropping (avoid recreation)
  private srcPoseCanvas: HTMLCanvasElement | null = null;
  private srcPoseCtx: CanvasRenderingContext2D | null = null;

  constructor(config: Pose3DDetectorConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    // Disable caching for large 3D models by default
    if (config.cache === undefined) {
      this.config.cache = false;
    }
  }

  /**
   * Initialize both detection and 3D pose models
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    try {
      // Load detection model
      console.log(`[Pose3DDetector] Loading detection model from: ${this.config.detModel}`);
      let detBuffer: ArrayBuffer;

      if (this.config.cache) {
        const detCached = await isModelCached(this.config.detModel);
        console.log(`[Pose3DDetector] Det model cache ${detCached ? 'hit' : 'miss'}`);
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
      console.log(`[Pose3DDetector] Detection model loaded, size: ${(detBuffer.byteLength / 1024 / 1024).toFixed(2)} MB`);

      // Load 3D pose model
      console.log(`[Pose3DDetector] Loading 3D pose model from: ${this.config.poseModel}`);
      let poseBuffer: ArrayBuffer;

      if (this.config.cache) {
        const poseCached = await isModelCached(this.config.poseModel);
        console.log(`[Pose3DDetector] 3D Pose model cache ${poseCached ? 'hit' : 'miss'}`);
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
      console.log(`[Pose3DDetector] 3D Pose model loaded, size: ${(poseBuffer.byteLength / 1024 / 1024).toFixed(2)} MB`);

      // Pre-allocate resources
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

      // Pose crop canvas
      this.poseCanvas = document.createElement('canvas');
      this.poseCanvas.width = poseW;
      this.poseCanvas.height = poseH;
      this.poseCtx = this.poseCanvas.getContext('2d', {
        willReadFrequently: true,
        alpha: false
      })!;

      // Pre-allocate pose tensor buffer
      this.poseTensorBuffer = new Float32Array(3 * poseW * poseH);

      // Source canvas will be created on first use (dynamic size)
      this.srcPoseCanvas = null;
      this.srcPoseCtx = null;

      this.initialized = true;
      console.log(`[Pose3DDetector] ✅ Initialized (det:${detW}x${detH}, pose:${poseW}x${poseH}, 3D)`);
    } catch (error) {
      console.error('[Pose3DDetector] ❌ Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Detect 3D poses from HTMLCanvasElement
   */
  async detectFromCanvas(canvas: HTMLCanvasElement): Promise<Wholebody3DResult> {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context from canvas');
    }

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return this.detect(new Uint8Array(imageData.data.buffer), canvas.width, canvas.height);
  }

  /**
   * Detect 3D poses from HTMLVideoElement
   */
  async detectFromVideo(
    video: HTMLVideoElement,
    targetCanvas?: HTMLCanvasElement
  ): Promise<Wholebody3DResult> {
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
   * Detect 3D poses from HTMLImageElement
   */
  async detectFromImage(
    image: HTMLImageElement,
    targetCanvas?: HTMLCanvasElement
  ): Promise<Wholebody3DResult> {
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
   * Detect 3D poses from ImageBitmap
   */
  async detectFromBitmap(
    bitmap: ImageBitmap,
    targetCanvas?: HTMLCanvasElement
  ): Promise<Wholebody3DResult> {
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
   * Detect 3D poses from File
   */
  async detectFromFile(
    file: File,
    targetCanvas?: HTMLCanvasElement
  ): Promise<Wholebody3DResult> {
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
   * Detect 3D poses from Blob
   */
  async detectFromBlob(
    blob: Blob,
    targetCanvas?: HTMLCanvasElement
  ): Promise<Wholebody3DResult> {
    const bitmap = await createImageBitmap(blob);
    const results = await this.detectFromBitmap(bitmap, targetCanvas);
    bitmap.close();
    return results;
  }

  /**
   * Detect 3D poses from raw image data
   */
  async detect(
    imageData: Uint8Array,
    width: number,
    height: number
  ): Promise<Wholebody3DResult> {
    if (!this.initialized) {
      await this.init();
    }

    const startTime = performance.now();

    // Step 1: Detect people
    const detStart = performance.now();
    const bboxes = await this.detectPeople(imageData, width, height);
    const detTime = performance.now() - detStart;

    // Step 2: Estimate 3D poses for each person
    const poseStart = performance.now();
    const allKeypoints: number[][][] = [];
    const allScores: number[][] = [];
    const allKeypointsSimcc: number[][][] = [];
    const allKeypoints2d: number[][][] = [];

    // Reset source canvas for new image (will be recreated on first bbox)
    this.srcPoseCanvas = null;
    this.srcPoseCtx = null;

    for (const bbox of bboxes) {
      const poseResult = await this.estimatePose3D(imageData, width, height, bbox);
      allKeypoints.push(poseResult.keypoints);
      allScores.push(poseResult.scores);
      allKeypointsSimcc.push(poseResult.keypointsSimcc);
      allKeypoints2d.push(poseResult.keypoints2d);
    }

    const poseTime = performance.now() - poseStart;
    const totalTime = performance.now() - startTime;

    // Attach stats
    const result: Wholebody3DResult = {
      keypoints: allKeypoints,
      scores: allScores,
      keypointsSimcc: allKeypointsSimcc,
      keypoints2d: allKeypoints2d,
    };

    (result as any).stats = {
      personCount: allKeypoints.length,
      detTime: Math.round(detTime),
      poseTime: Math.round(poseTime),
      totalTime: Math.round(totalTime),
    } as Pose3DStats;

    return result;
  }

  /**
   * Detect people using YOLOX
   */
  private async detectPeople(
    imageData: Uint8Array,
    width: number,
    height: number
  ): Promise<Array<{ x1: number; y1: number; x2: number; y2: number; confidence: number }>> {
    const [inputH, inputW] = this.config.detInputSize;

    const { tensor, paddingX, paddingY, scaleX, scaleY } = this.preprocessYOLO(
      imageData,
      width,
      height,
      [inputW, inputH]
    );

    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, inputH, inputW]);
    const inputName = this.detSession!.inputNames[0];

    const feeds: Record<string, ort.Tensor> = {};
    feeds[inputName] = inputTensor;

    const results = await this.detSession!.run(feeds);
    const output = results[this.detSession!.outputNames[0]];

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
   * Estimate 3D pose for a single person
   */
  private async estimatePose3D(
    imageData: Uint8Array,
    imgWidth: number,
    imgHeight: number,
    bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number }
  ): Promise<{
    keypoints: number[][];
    scores: number[];
    keypointsSimcc: number[][];
    keypoints2d: number[][];
  }> {
    const [inputW, inputH] = this.config.poseInputSize;

    const { tensor, center, scale } = this.preprocessPose(
      imageData,
      imgWidth,
      imgHeight,
      bbox,
      [inputW, inputH]
    );

    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, inputH, inputW]);
    
    // Use dynamic input name
    const inputName = this.poseSession!.inputNames[0];
    const feeds: Record<string, ort.Tensor> = {};
    feeds[inputName] = inputTensor;

    const results = await this.poseSession!.run(feeds);

    // Debug output names on first run only
    if (!this.outputNamesLogged) {
      console.log('[Pose3DDetector] Output names:', this.poseSession!.outputNames);
      console.log('[Pose3DDetector] Output shapes:', this.poseSession!.outputNames.map(k => results[k].dims));
      this.outputNamesLogged = true;
    }

    // Get output tensors using session's outputNames
    // Model input is [width=288, height=384], so:
    // X output has dim 576 (288*2), Y output has dim 768 (384*2)
    const outputNames = this.poseSession!.outputNames;
    let simccX: ort.Tensor, simccY: ort.Tensor, simccZ: ort.Tensor;

    // Find outputs by shape
    const shape0 = results[outputNames[0]].dims[2];
    const shape1 = results[outputNames[1]].dims[2];
    const shape2 = results[outputNames[2]].dims[2];

    // X has smaller shape (576), Y has larger (768)
    if (shape0 === 576) simccX = results[outputNames[0]];
    else if (shape1 === 576) simccX = results[outputNames[1]];
    else simccX = results[outputNames[2]];

    if (shape0 === 768) simccY = results[outputNames[0]];
    else if (shape1 === 768) simccY = results[outputNames[1]];
    else simccY = results[outputNames[2]];

    // Z is the remaining one
    const usedIndices = [
      simccX === results[outputNames[0]] ? 0 : simccX === results[outputNames[1]] ? 1 : 2,
      simccY === results[outputNames[0]] ? 0 : simccY === results[outputNames[1]] ? 1 : 2,
    ];
    simccZ = results[outputNames[3 - usedIndices[0] - usedIndices[1]]];

    return this.postprocessPose3D(
      simccX.data as Float32Array,
      simccY.data as Float32Array,
      simccZ.data as Float32Array,
      simccX.dims as number[],
      simccY.dims as number[],
      simccZ.dims as number[],
      center,
      scale,
      imgWidth,
      imgHeight
    );
  }

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

    if (!this.canvas || !this.ctx) {
      this.canvas = document.createElement('canvas');
      this.canvas.width = inputW;
      this.canvas.height = inputH;
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true, alpha: false })!;
    }

    const ctx = this.ctx;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, inputW, inputH);

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

    const srcCanvas = document.createElement('canvas');
    const srcCtx = srcCanvas.getContext('2d')!;
    srcCanvas.width = imgWidth;
    srcCanvas.height = imgHeight;

    const srcImageData = srcCtx.createImageData(imgWidth, imgHeight);
    srcImageData.data.set(imageData);
    srcCtx.putImageData(srcImageData, 0, 0);

    ctx.drawImage(srcCanvas, 0, 0, imgWidth, imgHeight, offsetX, offsetY, drawWidth, drawHeight);

    const paddedData = ctx.getImageData(0, 0, inputW, inputH);
    const tensor = new Float32Array(inputW * inputH * 3);

    for (let i = 0; i < paddedData.data.length; i += 4) {
      const pixelIdx = i / 4;
      tensor[pixelIdx] = paddedData.data[i] / 255;
      tensor[pixelIdx + inputW * inputH] = paddedData.data[i + 1] / 255;
      tensor[pixelIdx + 2 * inputW * inputH] = paddedData.data[i + 2] / 255;
    }

    const scaleX = imgWidth / drawWidth;
    const scaleY = imgHeight / drawHeight;

    return { tensor, paddingX: offsetX, paddingY: offsetY, scaleX, scaleY };
  }

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

    return this.applyNMS(detections, this.config.nmsThreshold);
  }

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

    // Center of bbox (same as Python)
    const center: [number, number] = [
      bbox.x1 + bboxWidth / 2,
      bbox.y1 + bboxHeight / 2,
    ];

    // Scale with padding (same as Python bbox_xyxy2cs with padding=1.25)
    let scaleW = bboxWidth * 1.25;
    let scaleH = bboxHeight * 1.25;

    // Adjust scale to match model aspect ratio (same as top_down_affine)
    const modelAspectRatio = inputW / inputH;
    const bboxAspectRatio = scaleW / scaleH;

    if (bboxAspectRatio > modelAspectRatio) {
      scaleH = scaleW / modelAspectRatio;
    } else {
      scaleW = scaleH * modelAspectRatio;
    }

    const scale: [number, number] = [scaleW, scaleH];

    // Reuse pose canvas
    if (!this.poseCanvas || !this.poseCtx) {
      this.poseCanvas = document.createElement('canvas');
      this.poseCanvas.width = inputW;
      this.poseCanvas.height = inputH;
      this.poseCtx = this.poseCanvas.getContext('2d', {
        willReadFrequently: true,
        alpha: false
      })!;
    }

    // Reuse source canvas for original image (avoid recreation per bbox)
    if (!this.srcPoseCanvas || !this.srcPoseCtx) {
      this.srcPoseCanvas = document.createElement('canvas');
      this.srcPoseCanvas.width = imgWidth;
      this.srcPoseCanvas.height = imgHeight;
      this.srcPoseCtx = this.srcPoseCanvas.getContext('2d', {
        willReadFrequently: true,
        alpha: false
      })!;
      // Copy image data once
      const srcImageData = this.srcPoseCtx.createImageData(imgWidth, imgHeight);
      srcImageData.data.set(imageData);
      this.srcPoseCtx.putImageData(srcImageData, 0, 0);
    }

    const ctx = this.poseCtx;
    ctx.clearRect(0, 0, inputW, inputH);

    // Crop and resize using drawImage (single GPU operation)
    const srcX = center[0] - scaleW / 2;
    const srcY = center[1] - scaleH / 2;
    ctx.drawImage(this.srcPoseCanvas, srcX, srcY, scaleW, scaleH, 0, 0, inputW, inputH);

    const croppedData = ctx.getImageData(0, 0, inputW, inputH);
    const tensor = this.poseTensorBuffer!;
    const len = croppedData.data.length;
    const planeSize = inputW * inputH;

    // Normalization constants
    const mean0 = 123.675, mean1 = 116.28, mean2 = 103.53;
    const stdInv0 = 1 / 58.395, stdInv1 = 1 / 57.12, stdInv2 = 1 / 57.375;

    // Optimized normalization loop - process 4 pixels at once (SIMD-like)
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

  private postprocessPose3D(
    simccX: Float32Array,
    simccY: Float32Array,
    simccZ: Float32Array,
    shapeX: number[],
    shapeY: number[],
    shapeZ: number[],
    center: [number, number],
    scale: [number, number],
    imgWidth: number,
    imgHeight: number
  ): {
    keypoints: number[][];
    scores: number[];
    keypointsSimcc: number[][];
    keypoints2d: number[][];
  } {
    const numKeypoints = shapeX[1];
    const wx = shapeX[2];
    const wy = shapeY[2];
    const wz = shapeZ[2];

    const keypoints: number[][] = [];
    const scores: number[] = [];
    const keypointsSimcc: number[][] = [];
    const keypoints2d: number[][] = [];

    for (let k = 0; k < numKeypoints; k++) {
      let maxX = -Infinity, argmaxX = 0;
      for (let i = 0; i < wx; i++) {
        const val = simccX[k * wx + i];
        if (val > maxX) { maxX = val; argmaxX = i; }
      }

      let maxY = -Infinity, argmaxY = 0;
      for (let i = 0; i < wy; i++) {
        const val = simccY[k * wy + i];
        if (val > maxY) { maxY = val; argmaxY = i; }
      }

      let maxZ = -Infinity, argmaxZ = 0;
      for (let i = 0; i < wz; i++) {
        const val = simccZ[k * wz + i];
        if (val > maxZ) { maxZ = val; argmaxZ = i; }
      }

      const score = maxX > maxY ? maxX : maxY;
      
      // Normalize to [0, 1]
      const normX = argmaxX / wx;
      const normY = argmaxY / wy;
      const normZ = argmaxZ / wz;

      // 3D coordinates in model space
      const kptX = (normX - 0.5) * 2.0;
      const kptY = (normY - 0.5) * 2.0;
      const kptZMetric = (normZ - 0.5) * this.config.zRange * 2;

      keypoints.push([kptX, kptY, kptZMetric]);
      keypointsSimcc.push([normX, normY, normZ]);

      // 2D coordinates in original image space
      // Convert from normalized SimCC coords [0, 1] to crop space, then to image space
      // Formula: kpt = center - scale/2 + norm * scale (same as in rtmpose3d.ts)
      const kpt2dX = normX * scale[0] + center[0] - 0.5 * scale[0];
      const kpt2dY = normY * scale[1] + center[1] - 0.5 * scale[1];

      // Clamp to image bounds
      const clampedX = Math.max(0, Math.min(imgWidth, kpt2dX));
      const clampedY = Math.max(0, Math.min(imgHeight, kpt2dY));

      keypoints2d.push([clampedX, clampedY]);

      scores.push(score);
    }

    return { keypoints, scores, keypointsSimcc, keypoints2d };
  }

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
}
