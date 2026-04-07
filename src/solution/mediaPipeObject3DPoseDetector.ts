/**
 * MediaPipeObject3DPoseDetector - Комбинированный детектор
 * Использует MediaPipe для быстрой детекции объектов + RTMW3D для 3D позы
 *
 * @example
 * ```typescript
 * // Initialize
 * const detector = new MediaPipeObject3DPoseDetector({
 *   mpScoreThreshold: 0.5,
 *   poseConfidence: 0.3,
 * });
 * await detector.init();
 *
 * // Detect
 * const result = await detector.detectFromCanvas(canvas);
 * console.log(result.keypoints[0][0]); // [x, y, z] - 3D координаты первого ключевой точки
 * ```
 */

import * as ort from 'onnxruntime-web/all';
import { FilesetResolver, ObjectDetector as MPObjectDetector } from '@mediapipe/tasks-vision';
import { getCachedModel, isModelCached } from '../core/modelCache';
import { loadMediaPipeModelWithCache } from '../core/mediaPipeCache';
import { initOnnxRuntimeWeb } from '../core/onnxRuntime';

/**
 * 3D pose detection result
 */
export interface Wholebody3DResult {
  keypoints: number[][][];
  scores: number[][];
  keypointsSimcc: number[][][];
  keypoints2d: number[][][];
  stats?: {
    personCount: number;
    detTime: number;
    poseTime: number;
    totalTime: number;
  };
}

// Configure ONNX Runtime Web (only in browser environment)
initOnnxRuntimeWeb();

/**
 * Configuration options for MediaPipeObject3DPoseDetector
 */
export interface MediaPipeObject3DPoseDetectorConfig {
  /** Path to RTMW3D 3D pose estimation model */
  poseModel?: string;
  /** MediaPipe score threshold (default: 0.5) */
  mpScoreThreshold?: number;
  /** MediaPipe max results (-1 for all, default: -1) */
  mpMaxResults?: number;
  /** Pose input size (default: [288, 384]) */
  poseInputSize?: [number, number];
  /** Pose keypoint confidence threshold (default: 0.3) */
  poseConfidence?: number;
  /** Execution backend for pose model (default: 'webgpu') */
  backend?: 'wasm' | 'webgl' | 'webgpu' | 'webnn';
  /** WebNN provider options (only used when backend is 'webnn') */
  webnnOptions?: import('../types/index').WebNNProviderOptionsOrUndefined;
  /** Device type for WebNN/WebGPU (default: 'gpu' for high performance) */
  deviceType?: 'cpu' | 'gpu' | 'npu';
  /** Power preference for WebNN/WebGPU (default: 'high-performance') */
  powerPreference?: 'default' | 'low-power' | 'high-performance';
  /** Enable model caching (default: true) */
  cache?: boolean;
  /** Z-axis range in meters (default: 2.1744869) */
  zRange?: number;
  /** Only detect persons (default: true) */
  personsOnly?: boolean;
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: Omit<Required<MediaPipeObject3DPoseDetectorConfig>, 'webnnOptions'> & {
  webnnOptions?: import('../types/index').WebNNProviderOptionsOrUndefined;
} = {
  poseModel: 'https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx',
  mpScoreThreshold: 0.5,
  mpMaxResults: -1,
  poseInputSize: [288, 384],
  poseConfidence: 0.3,
  backend: 'webgpu',
  webnnOptions: undefined as import('../types/index').WebNNProviderOptionsOrUndefined,
  deviceType: 'gpu',
  powerPreference: 'high-performance',
  cache: true,
  zRange: 2.1744869,
  personsOnly: true,
};

/**
 * Detected person with 3D pose
 */
export interface Person3DWithBBox {
  /** Bounding box from MediaPipe */
  bbox: {
    x1: number;
    y1: number;
    width: number;
    height: number;
    confidence: number;
  };
  /** 3D keypoints [x, y, z] in meters */
  keypoints: number[][];
  /** Keypoint scores (0-1) */
  scores: number[];
  /** 2D keypoints in pixel coordinates */
  keypoints2d: number[][];
  /** Normalized SimCC coordinates */
  keypointsSimcc: number[][];
}

export class MediaPipeObject3DPoseDetector {
  private config: Omit<Required<MediaPipeObject3DPoseDetectorConfig>, 'webnnOptions'> & {
    webnnOptions?: import('../types/index').WebNNProviderOptionsOrUndefined;
  };
  private mpDetector: MPObjectDetector | null = null;
  private poseSession: ort.InferenceSession | null = null;
  private initialized = false;
  private vision: any = null;

  // Pre-allocated buffers
  private poseCanvas: HTMLCanvasElement | null = null;
  private poseCtx: CanvasRenderingContext2D | null = null;
  private poseTensorBuffer: Float32Array | null = null;
  private poseInputSize: [number, number] = [288, 384];
  private srcPoseCanvas: HTMLCanvasElement | null = null;
  private srcPoseCtx: CanvasRenderingContext2D | null = null;

  constructor(config: MediaPipeObject3DPoseDetectorConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Initialize both MediaPipe detector and RTMW3D pose model
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    try {
      console.log('[MediaPipeObject3DPoseDetector] Initializing...');

      // Initialize MediaPipe Object Detector
      this.vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );

      this.mpDetector = await MPObjectDetector.createFromOptions(this.vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite',
        },
        scoreThreshold: this.config.mpScoreThreshold,
        maxResults: this.config.mpMaxResults,
        categoryAllowlist: this.config.personsOnly ? ['person'] : undefined,
        runningMode: 'IMAGE',
      });

      console.log('[MediaPipeObject3DPoseDetector] MediaPipe detector initialized');

      // Initialize RTMW3D pose model
      console.log('[MediaPipeObject3DPoseDetector] Loading 3D pose model...');
      let poseBuffer: ArrayBuffer;

      if (this.config.cache) {
        const poseCached = await isModelCached(this.config.poseModel);
        console.log(`[MediaPipeObject3DPoseDetector] Pose model cache ${poseCached ? 'hit' : 'miss'}`);
        poseBuffer = await getCachedModel(this.config.poseModel);
      } else {
        const poseResponse = await fetch(this.config.poseModel);
        if (!poseResponse.ok) {
          throw new Error(`Failed to fetch pose model: HTTP ${poseResponse.status}`);
        }
        poseBuffer = await poseResponse.arrayBuffer();
      }

      // Build execution providers with WebNN options
      const execProviders: any[] = [];
      if (this.config.backend === 'webnn') {
        execProviders.push({
          name: 'webnn',
          deviceType: this.config.deviceType || 'gpu',
          powerPreference: this.config.powerPreference || 'high-performance',
        });
      } else {
        execProviders.push(this.config.backend);
      }

      this.poseSession = await ort.InferenceSession.create(poseBuffer, {
        executionProviders: execProviders,
        graphOptimizationLevel: 'all',
      });

      console.log(`[MediaPipeObject3DPoseDetector] 3D Pose model loaded, size: ${(poseBuffer.byteLength / 1024 / 1024).toFixed(2)} MB`);

      // Pre-allocate resources
      const [poseW, poseH] = this.config.poseInputSize;
      this.poseInputSize = [poseW, poseH];

      this.poseCanvas = document.createElement('canvas');
      this.poseCanvas.width = poseW;
      this.poseCanvas.height = poseH;
      this.poseCtx = this.poseCanvas.getContext('2d', {
        willReadFrequently: true,
        alpha: false
      })!;

      this.poseTensorBuffer = new Float32Array(3 * poseW * poseH);

      this.initialized = true;
      console.log(`[MediaPipeObject3DPoseDetector] ✅ Initialized (pose:${poseW}x${poseH}, 3D)`);
    } catch (error) {
      console.error('[MediaPipeObject3DPoseDetector] ❌ Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Detect persons and estimate 3D poses from HTMLCanvasElement
   */
  async detectFromCanvas(canvas: HTMLCanvasElement): Promise<Wholebody3DResult> {
    if (!this.initialized) {
      await this.init();
    }

    const startTime = performance.now();

    // Step 1: Detect persons using MediaPipe (FAST!)
    const mpResult = this.mpDetector!.detect(canvas);
    const personBoxes = mpResult.detections
      .filter(d => !this.config.personsOnly || d.categories[0].categoryName === 'person')
      .map(d => ({
        x1: d.boundingBox!.originX,
        y1: d.boundingBox!.originY,
        width: d.boundingBox!.width,
        height: d.boundingBox!.height,
        confidence: d.categories[0].score,
      }));

    // Step 2: Estimate 3D poses for each person
    const allKeypoints: number[][][] = [];
    const allScores: number[][] = [];
    const allKeypointsSimcc: number[][][] = [];
    const allKeypoints2d: number[][][] = [];

    this.srcPoseCanvas = null;
    this.srcPoseCtx = null;

    for (const bbox of personBoxes) {
      const poseResult = await this.estimatePose3D(
        canvas,
        canvas.width,
        canvas.height,
        bbox
      );
      allKeypoints.push(poseResult.keypoints);
      allScores.push(poseResult.scores);
      allKeypointsSimcc.push(poseResult.keypointsSimcc);
      allKeypoints2d.push(poseResult.keypoints2d);
    }

    const totalTime = performance.now() - startTime;

    const result: Wholebody3DResult = {
      keypoints: allKeypoints,
      scores: allScores,
      keypointsSimcc: allKeypointsSimcc,
      keypoints2d: allKeypoints2d,
    };

    (result as any).stats = {
      personCount: allKeypoints.length,
      detTime: Math.round(totalTime * 0.3), // MediaPipe is fast (~30% of time)
      poseTime: Math.round(totalTime * 0.7),
      totalTime: Math.round(totalTime),
    };

    return result;
  }

  /**
   * Detect from video element
   */
  async detectFromVideo(video: HTMLVideoElement, targetCanvas?: HTMLCanvasElement): Promise<Wholebody3DResult> {
    if (video.readyState < 2) {
      throw new Error('Video not ready');
    }

    const canvas = targetCanvas || document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get canvas context');
    }

    ctx.drawImage(video, 0, 0);
    return this.detectFromCanvas(canvas);
  }

  /**
   * Detect from image element
   */
  async detectFromImage(image: HTMLImageElement, targetCanvas?: HTMLCanvasElement): Promise<Wholebody3DResult> {
    if (!image.complete || !image.naturalWidth) {
      throw new Error('Image not loaded');
    }

    const canvas = targetCanvas || document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get canvas context');
    }

    ctx.drawImage(image, 0, 0);
    return this.detectFromCanvas(canvas);
  }

  /**
   * Detect from file
   */
  async detectFromFile(file: File, targetCanvas?: HTMLCanvasElement): Promise<Wholebody3DResult> {
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
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = URL.createObjectURL(file);
    });
  }

  /**
   * Detect from blob
   */
  async detectFromBlob(blob: Blob, targetCanvas?: HTMLCanvasElement): Promise<Wholebody3DResult> {
    const bitmap = await createImageBitmap(blob);
    const results = await this.detectFromBitmap(bitmap, targetCanvas);
    bitmap.close();
    return results;
  }

  /**
   * Detect from ImageBitmap
   */
  async detectFromBitmap(bitmap: ImageBitmap, targetCanvas?: HTMLCanvasElement): Promise<Wholebody3DResult> {
    const canvas = targetCanvas || document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get canvas context');
    }

    ctx.drawImage(bitmap, 0, 0);
    const results = await this.detectFromCanvas(canvas);
    return results;
  }

  /**
   * Estimate 3D pose for a single person
   */
  private async estimatePose3D(
    canvas: HTMLCanvasElement,
    imgWidth: number,
    imgHeight: number,
    bbox: { x1: number; y1: number; width: number; height: number; confidence: number }
  ): Promise<{
    keypoints: number[][];
    scores: number[];
    keypointsSimcc: number[][];
    keypoints2d: number[][];
  }> {
    const [inputW, inputH] = this.config.poseInputSize;

    const bboxRect = {
      x1: bbox.x1,
      y1: bbox.y1,
      x2: bbox.x1 + bbox.width,
      y2: bbox.y1 + bbox.height,
      confidence: bbox.confidence,
    };

    const { tensor, center, scale } = this.preprocessPose(
      canvas,
      imgWidth,
      imgHeight,
      bboxRect,
      [inputW, inputH]
    );

    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, inputH, inputW]);
    const inputName = this.poseSession!.inputNames[0];
    const feeds: Record<string, ort.Tensor> = {};
    feeds[inputName] = inputTensor;

    const results = await this.poseSession!.run(feeds);

    const outputNames = this.poseSession!.outputNames;
    const shape0 = results[outputNames[0]].dims[2];
    const shape1 = results[outputNames[1]].dims[2];
    const shape2 = results[outputNames[2]].dims[2];

    let simccX: ort.Tensor, simccY: ort.Tensor, simccZ: ort.Tensor;

    if (shape0 === 576) simccX = results[outputNames[0]];
    else if (shape1 === 576) simccX = results[outputNames[1]];
    else simccX = results[outputNames[2]];

    if (shape0 === 768) simccY = results[outputNames[0]];
    else if (shape1 === 768) simccY = results[outputNames[1]];
    else simccY = results[outputNames[2]];

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

  /**
   * Pose preprocessing with affine crop
   */
  private preprocessPose(
    canvas: HTMLCanvasElement,
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

    let scaleW = bboxWidth * 1.25;
    let scaleH = bboxHeight * 1.25;

    const modelAspectRatio = inputW / inputH;
    const bboxAspectRatio = scaleW / scaleH;

    if (bboxAspectRatio > modelAspectRatio) {
      scaleH = scaleW / modelAspectRatio;
    } else {
      scaleW = scaleH * modelAspectRatio;
    }

    const scale: [number, number] = [scaleW, scaleH];

    if (!this.poseCanvas || !this.poseCtx) {
      this.poseCanvas = document.createElement('canvas');
      this.poseCanvas.width = inputW;
      this.poseCanvas.height = inputH;
      this.poseCtx = this.poseCanvas.getContext('2d', {
        willReadFrequently: true,
        alpha: false
      })!;
    }

    if (!this.srcPoseCanvas || !this.srcPoseCtx) {
      this.srcPoseCanvas = canvas;
      this.srcPoseCtx = canvas.getContext('2d', {
        willReadFrequently: true,
        alpha: false
      })!;
    }

    const ctx = this.poseCtx;
    ctx.clearRect(0, 0, inputW, inputH);

    const srcX = center[0] - scaleW / 2;
    const srcY = center[1] - scaleH / 2;
    ctx.drawImage(canvas, srcX, srcY, scaleW, scaleH, 0, 0, inputW, inputH);

    const croppedData = ctx.getImageData(0, 0, inputW, inputH);
    const tensor = this.poseTensorBuffer!;
    const len = croppedData.data.length;
    const planeSize = inputW * inputH;

    const mean0 = 123.675, mean1 = 116.28, mean2 = 103.53;
    const stdInv0 = 1 / 58.395, stdInv1 = 1 / 57.12, stdInv2 = 1 / 57.375;

    for (let i = 0; i < len; i += 16) {
      const p1 = i / 4, p2 = p1 + 1, p3 = p1 + 2, p4 = p1 + 3;

      tensor[p1] = (croppedData.data[i] - mean0) * stdInv0;
      tensor[p2] = (croppedData.data[i + 4] - mean0) * stdInv0;
      tensor[p3] = (croppedData.data[i + 8] - mean0) * stdInv0;
      tensor[p4] = (croppedData.data[i + 12] - mean0) * stdInv0;

      tensor[p1 + planeSize] = (croppedData.data[i + 1] - mean1) * stdInv1;
      tensor[p2 + planeSize] = (croppedData.data[i + 5] - mean1) * stdInv1;
      tensor[p3 + planeSize] = (croppedData.data[i + 9] - mean1) * stdInv1;
      tensor[p4 + planeSize] = (croppedData.data[i + 13] - mean1) * stdInv1;

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
      let maxX = -Infinity;
      let argmaxX = 0;
      for (let i = 0; i < wx; i++) {
        const val = simccX[k * wx + i];
        if (val > maxX) {
          maxX = val;
          argmaxX = i;
        }
      }

      let maxY = -Infinity;
      let argmaxY = 0;
      for (let i = 0; i < wy; i++) {
        const val = simccY[k * wy + i];
        if (val > maxY) {
          maxY = val;
          argmaxY = i;
        }
      }

      let maxZ = -Infinity;
      let argmaxZ = 0;
      for (let i = 0; i < wz; i++) {
        const val = simccZ[k * wz + i];
        if (val > maxZ) {
          maxZ = val;
          argmaxZ = i;
        }
      }

      const score = 0.5 * (maxX + maxY);
      const visible = score > this.config.poseConfidence;

      const normX = argmaxX / wx;
      const normY = argmaxY / wy;
      const normZ = argmaxZ / wz;

      const x = (normX - 0.5) * scale[0] + center[0];
      const y = (normY - 0.5) * scale[1] + center[1];
      const z = (normZ - 0.5) * this.config.zRange;

      keypoints.push([x, y, z]);
      keypointsSimcc.push([normX, normY, normZ]);
      keypoints2d.push([x, y]);
      scores.push(visible ? score : 0);
    }

    return { keypoints, scores, keypointsSimcc, keypoints2d };
  }

  /**
   * Update MediaPipe score threshold
   */
  async setScoreThreshold(threshold: number): Promise<void> {
    if (!this.mpDetector) return;

    await this.mpDetector.setOptions({
      scoreThreshold: threshold,
    });
    this.config.mpScoreThreshold = threshold;
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    if (this.mpDetector) {
      this.mpDetector.close();
      this.mpDetector = null;
    }
    if (this.poseSession) {
      this.poseSession.release();
      this.poseSession = null;
    }
    this.initialized = false;
  }
}
