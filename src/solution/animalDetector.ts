/**
 * AnimalDetector - Animal detection and pose estimation API
 * Supports 30 animal classes with ViTPose++ pose model
 *
 * @example
 * ```typescript
 * // Initialize with default models
 * const detector = new AnimalDetector();
 * await detector.init();
 *
 * // Detect animals
 * const animals = await detector.detectFromCanvas(canvas);
 * console.log(`Found ${animals.length} animals`);
 *
 * // With custom models
 * const detector2 = new AnimalDetector({
 *   detModel: 'path/to/yolox_animal.onnx',
 *   poseModel: 'path/to/vitpose_animal.onnx',
 * });
 * ```
 */

import * as ort from 'onnxruntime-web';
import { getCachedModel, isModelCached } from '../core/modelCache';

// Configure ONNX Runtime Web
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';
ort.env.wasm.simd = true;
ort.env.wasm.proxy = false;

/**
 * 30 Animal class names supported by AnimalDetector
 */
export const ANIMAL_CLASSES: string[] = [
  'gorilla',
  'spider-monkey',
  'howling-monkey',
  'zebra',
  'elephant',
  'hippo',
  'raccon',
  'rhino',
  'giraffe',
  'tiger',
  'deer',
  'lion',
  'panda',
  'cheetah',
  'black-bear',
  'polar-bear',
  'antelope',
  'fox',
  'buffalo',
  'cow',
  'wolf',
  'dog',
  'sheep',
  'cat',
  'horse',
  'rabbit',
  'pig',
  'chimpanzee',
  'monkey',
  'orangutan',
];

/**
 * Available ViTPose++ models for animal pose estimation
 * All models are trained on 6 datasets and support 30 animal classes
 */
export const VITPOSE_MODELS = {
  /** ViTPose++-s: Fastest, 74.2 AP on AP10K */
  'vitpose-s': {
    name: 'ViTPose++-s',
    url: 'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-s-apt36k.onnx',
    inputSize: [256, 192] as [number, number],
    ap: 74.2,
    description: 'Fastest inference, suitable for real-time applications',
  },
  /** ViTPose++-b: Balanced, 75.9 AP on AP10K */
  'vitpose-b': {
    name: 'ViTPose++-b',
    url: 'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-b-apt36k.onnx',
    inputSize: [256, 192] as [number, number],
    ap: 75.9,
    description: 'Balanced speed and accuracy',
  },
  /** ViTPose++-l: Most accurate, 80.8 AP on AP10K */
  'vitpose-l': {
    name: 'ViTPose++-l',
    url: 'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-h-apt36k.onnx',
    inputSize: [256, 192] as [number, number],
    ap: 80.8,
    description: 'Highest accuracy, slower inference',
  },
} as const;

export type VitPoseModelType = keyof typeof VITPOSE_MODELS;

/**
 * Configuration options for AnimalDetector
 */
export interface AnimalDetectorConfig {
  /** Path to animal detection model (optional - uses default if not specified) */
  detModel?: string;
  /** Path to animal pose estimation model (optional - uses default if not specified) */
  poseModel?: string;
  /** ViTPose++ model variant (optional - uses vitpose-b if not specified) */
  poseModelType?: VitPoseModelType;
  /** Detection input size (default: [640, 640]) */
  detInputSize?: [number, number];
  /** Pose input size (default: [256, 192]) */
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
  /** Animal classes to detect (null = all) */
  classes?: string[] | null;
}

/**
 * Detected animal with bounding box and keypoints
 */
export interface DetectedAnimal {
  /** Bounding box coordinates */
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
  };
  /** Animal class ID */
  classId: number;
  /** Animal class name */
  className: string;
  /** 17 keypoints (COCO format) */
  keypoints: AnimalKeypoint[];
  /** Keypoint scores (0-1) */
  scores: number[];
}

/**
 * Single keypoint with coordinates and visibility
 */
export interface AnimalKeypoint {
  x: number;
  y: number;
  score: number;
  visible: boolean;
  name: string;
}

/**
 * Detection statistics
 */
export interface AnimalDetectionStats {
  /** Number of detected animals */
  animalCount: number;
  /** Detections per class */
  classCounts: Record<string, number>;
  /** Detection inference time (ms) */
  detTime: number;
  /** Pose estimation time (ms) */
  poseTime: number;
  /** Total processing time (ms) */
  totalTime: number;
}

/**
 * COCO17 keypoint names (used for animal pose)
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
 * Default configuration - uses ViTPose++-b model
 */
const DEFAULT_CONFIG: Required<Omit<AnimalDetectorConfig, 'poseModel' | 'poseModelType'>> & {
  poseModel?: string;
  poseModelType: VitPoseModelType;
} = {
  detModel: 'https://huggingface.co/demon2233/rtmlib-ts/resolve/main/yolo/yolov12n.onnx',
  poseModel: undefined,  // Will be set from poseModelType
  poseModelType: 'vitpose-b',
  detInputSize: [640, 640],
  poseInputSize: [256, 192],
  detConfidence: 0.5,
  nmsThreshold: 0.45,
  poseConfidence: 0.3,
  backend: 'webgpu',  // Default to WebGPU for better performance
  cache: true,
  classes: null,
};

export class AnimalDetector {
  private config: Required<AnimalDetectorConfig>;
  private detSession: ort.InferenceSession | null = null;
  private poseSession: ort.InferenceSession | null = null;
  private initialized = false;
  private classFilter: Set<number> | null = null;

  // Pre-allocated buffers
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private poseCanvas: HTMLCanvasElement | null = null;
  private poseCtx: CanvasRenderingContext2D | null = null;
  private poseTensorBuffer: Float32Array | null = null;
  private detInputSize: [number, number] = [640, 640];
  private poseInputSize: [number, number] = [256, 192];

  constructor(config: AnimalDetectorConfig = {}) {
    // Resolve pose model URL from poseModelType if poseModel not explicitly provided
    let finalConfig = { ...DEFAULT_CONFIG, ...config };
    
    if (!config.poseModel && config.poseModelType) {
      const vitposeModel = VITPOSE_MODELS[config.poseModelType];
      finalConfig.poseModel = vitposeModel.url;
      finalConfig.poseInputSize = vitposeModel.inputSize;
    } else if (!config.poseModel && !config.poseModelType) {
      // Use default vitpose-b
      finalConfig.poseModel = VITPOSE_MODELS['vitpose-b'].url;
      finalConfig.poseInputSize = VITPOSE_MODELS['vitpose-b'].inputSize;
    }
    
    this.config = finalConfig as Required<AnimalDetectorConfig>;
    this.updateClassFilter();
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
      const classId = ANIMAL_CLASSES.indexOf(className.toLowerCase());
      if (classId !== -1) {
        this.classFilter!.add(classId);
      } else {
        console.warn(`[AnimalDetector] Unknown class: ${className}`);
      }
    });
  }

  /**
   * Set which animal classes to detect
   */
  setClasses(classes: string[] | null): void {
    this.config.classes = classes;
    this.updateClassFilter();
  }

  /**
   * Get list of available animal classes
   */
  getAvailableClasses(): string[] {
    return [...ANIMAL_CLASSES];
  }

  /**
   * Get information about the current ViTPose++ model
   */
  getPoseModelInfo() {
    const modelType = (this.config as any).poseModelType as VitPoseModelType;
    if (modelType && VITPOSE_MODELS[modelType]) {
      return VITPOSE_MODELS[modelType];
    }
    return null;
  }

  /**
   * Initialize both detection and pose models
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    try {
      // Load detection model
      console.log(`[AnimalDetector] Loading detection model from: ${this.config.detModel}`);
      let detBuffer: ArrayBuffer;

      if (this.config.cache) {
        const detCached = await isModelCached(this.config.detModel);
        console.log(`[AnimalDetector] Det model cache ${detCached ? 'hit' : 'miss'}`);
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
      console.log(`[AnimalDetector] Detection model loaded, size: ${(detBuffer.byteLength / 1024 / 1024).toFixed(2)} MB`);

      // Load pose model
      console.log(`[AnimalDetector] Loading pose model from: ${this.config.poseModel}`);
      let poseBuffer: ArrayBuffer;

      if (this.config.cache) {
        const poseCached = await isModelCached(this.config.poseModel);
        console.log(`[AnimalDetector] Pose model cache ${poseCached ? 'hit' : 'miss'}`);
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
      console.log(`[AnimalDetector] Pose model loaded, size: ${(poseBuffer.byteLength / 1024 / 1024).toFixed(2)} MB`);

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

      this.initialized = true;
      console.log(`[AnimalDetector] ✅ Initialized (det:${detW}x${detH}, pose:${poseW}x${poseH})`);
    } catch (error) {
      console.error('[AnimalDetector] ❌ Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Detect animals from HTMLCanvasElement
   */
  async detectFromCanvas(canvas: HTMLCanvasElement): Promise<DetectedAnimal[]> {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context from canvas');
    }

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return this.detect(new Uint8Array(imageData.data.buffer), canvas.width, canvas.height);
  }

  /**
   * Detect animals from HTMLVideoElement
   */
  async detectFromVideo(
    video: HTMLVideoElement,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectedAnimal[]> {
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
   * Detect animals from HTMLImageElement
   */
  async detectFromImage(
    image: HTMLImageElement,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectedAnimal[]> {
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
   * Detect animals from ImageBitmap
   */
  async detectFromBitmap(
    bitmap: ImageBitmap,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectedAnimal[]> {
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
   * Detect animals from File
   */
  async detectFromFile(
    file: File,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectedAnimal[]> {
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
   * Detect animals from Blob
   */
  async detectFromBlob(
    blob: Blob,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectedAnimal[]> {
    const bitmap = await createImageBitmap(blob);
    const results = await this.detectFromBitmap(bitmap, targetCanvas);
    bitmap.close();
    return results;
  }

  /**
   * Detect animals from raw image data
   */
  async detect(
    imageData: Uint8Array,
    width: number,
    height: number
  ): Promise<DetectedAnimal[]> {
    if (!this.initialized) {
      await this.init();
    }

    const startTime = performance.now();

    // Step 1: Detect animals
    const detStart = performance.now();
    const detections = await this.detectAnimals(imageData, width, height);
    const detTime = performance.now() - detStart;

    // Step 2: Estimate poses for each animal
    const poseStart = performance.now();
    const animals: DetectedAnimal[] = [];

    for (const det of detections) {
      const keypoints = await this.estimatePose(imageData, width, height, det.bbox);
      animals.push({
        bbox: det.bbox,
        classId: det.classId,
        className: det.className,
        keypoints,
        scores: keypoints.map(k => k.score),
      });
    }

    const poseTime = performance.now() - poseStart;
    const totalTime = performance.now() - startTime;

    // Calculate stats
    const classCounts: Record<string, number> = {};
    animals.forEach(animal => {
      classCounts[animal.className] = (classCounts[animal.className] || 0) + 1;
    });

    // Attach stats
    (animals as any).stats = {
      animalCount: animals.length,
      classCounts,
      detTime: Math.round(detTime),
      poseTime: Math.round(poseTime),
      totalTime: Math.round(totalTime),
    } as AnimalDetectionStats;

    return animals;
  }

  /**
   * Detect animals using YOLO
   */
  private async detectAnimals(
    imageData: Uint8Array,
    width: number,
    height: number
  ): Promise<Array<{
    bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number };
    classId: number;
    className: string;
  }>> {
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
   * Estimate pose for a single animal
   */
  private async estimatePose(
    imageData: Uint8Array,
    imgWidth: number,
    imgHeight: number,
    bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number }
  ): Promise<AnimalKeypoint[]> {
    const [inputH, inputW] = this.config.poseInputSize;

    const { tensor, center, scale } = this.preprocessPose(
      imageData,
      imgWidth,
      imgHeight,
      bbox,
      [inputW, inputH]
    );

    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, inputH, inputW]);
    const results = await this.poseSession!.run({ input: inputTensor });

    return this.postprocessPose(
      results.simcc_x.data as Float32Array,
      results.simcc_y.data as Float32Array,
      results.simcc_x.dims as number[],
      results.simcc_y.dims as number[],
      center,
      scale
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
  ): Array<{
    bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number };
    classId: number;
    className: string;
  }> {
    const detections: Array<{
      bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number };
      classId: number;
      className: string;
    }> = [];

    for (let i = 0; i < numDetections; i++) {
      const idx = i * 6;
      const x1 = output[idx];
      const y1 = output[idx + 1];
      const x2 = output[idx + 2];
      const y2 = output[idx + 3];
      const confidence = output[idx + 4];
      const classId = Math.round(output[idx + 5]);

      if (confidence < this.config.detConfidence) continue;
      if (this.classFilter && !this.classFilter.has(classId)) continue;

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
        className: ANIMAL_CLASSES[classId] || `animal_${classId}`,
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

    const center: [number, number] = [
      bbox.x1 + bboxWidth / 2,
      bbox.y1 + bboxHeight / 2,
    ];

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

    if (!this.poseCanvas || !this.poseCtx) {
      this.poseCanvas = document.createElement('canvas');
      this.poseCanvas.width = inputW;
      this.poseCanvas.height = inputH;
      this.poseCtx = this.poseCanvas.getContext('2d', { willReadFrequently: true, alpha: false })!;
      this.poseTensorBuffer = new Float32Array(3 * inputW * inputH);
    }

    const ctx = this.poseCtx;
    ctx.clearRect(0, 0, inputW, inputH);

    const srcCanvas = document.createElement('canvas');
    const srcCtx = srcCanvas.getContext('2d')!;
    srcCanvas.width = imgWidth;
    srcCanvas.height = imgHeight;

    const srcImageData = srcCtx.createImageData(imgWidth, imgHeight);
    srcImageData.data.set(imageData);
    srcCtx.putImageData(srcImageData, 0, 0);

    const srcX = center[0] - scaleW / 2;
    const srcY = center[1] - scaleH / 2;
    ctx.drawImage(srcCanvas, srcX, srcY, scaleW, scaleH, 0, 0, inputW, inputH);

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

  private postprocessPose(
    simccX: Float32Array,
    simccY: Float32Array,
    shapeX: number[],
    shapeY: number[],
    center: [number, number],
    scale: [number, number]
  ): AnimalKeypoint[] {
    const numKeypoints = shapeX[1];
    const wx = shapeX[2];
    const wy = shapeY[2];

    const keypoints: AnimalKeypoint[] = [];

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

      const score = 0.5 * (maxX + maxY);
      const visible = score > this.config.poseConfidence;

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

  private applyNMS<T extends { bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number } }>(
    detections: T[],
    iouThreshold: number
  ): T[] {
    if (detections.length === 0) return [];

    detections.sort((a, b) => b.bbox.confidence - a.bbox.confidence);

    const selected: T[] = [];
    const used = new Set<number>();

    for (let i = 0; i < detections.length; i++) {
      if (used.has(i)) continue;

      selected.push(detections[i]);
      used.add(i);

      for (let j = i + 1; j < detections.length; j++) {
        if (used.has(j)) continue;

        const iou = this.calculateIoU(detections[i].bbox, detections[j].bbox);
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
