/**
 * CustomDetector - Maximum flexibility detector for any ONNX model
 * Provides low-level API for custom model inference
 *
 * @example
 * ```typescript
 * // Simple usage with auto-config
 * const detector = new CustomDetector({
 *   model: 'path/to/model.onnx',
 * });
 * await detector.init();
 * const results = await detector.run(imageData, width, height);
 *
 * // Advanced usage with custom preprocessing
 * const detector = new CustomDetector({
 *   model: 'path/to/model.onnx',
 *   inputName: 'input',
 *   outputNames: ['output1', 'output2'],
 *   preprocessing: (data) => customPreprocess(data),
 *   postprocessing: (outputs) => customPostprocess(outputs),
 * });
 * ```
 */

import * as ort from 'onnxruntime-web/all';
import { initOnnxRuntimeWeb } from '../core/onnxRuntime';
import { loadOnnxSession } from '../core/onnxSession';
import { createLogger } from '../core/logger';
import { loadBitmapFromBlob, loadImageFromFile } from '../core/sourceLoaders';
import { letterboxGeometry, fillLetterbox, rgbaToCHW } from '../core/preprocessing';
import type { WebNNProviderOptions } from '../types/index';

// Configure ONNX Runtime Web (only in browser environment)
initOnnxRuntimeWeb();

const log = createLogger('CustomDetector');

/**
 * Configuration options for CustomDetector
 */
export interface CustomDetectorConfig {
  /** Path to ONNX model (required) */
  model: string;
  /** Input tensor name (optional - auto-detected if not specified) */
  inputName?: string;
  /** Output tensor names (optional - auto-detected if not specified) */
  outputNames?: string[];
  /** Expected input shape [batch, channels, height, width] (optional) */
  inputShape?: [number, number, number, number];
  /** Custom preprocessing function */
  preprocessing?: (data: ImageData, config: CustomDetectorConfig) => Float32Array | ort.Tensor;
  /** Custom postprocessing function */
  postprocessing?: (outputs: Record<string, ort.Tensor>, metadata: any) => any;
  /** Execution backend (default: 'wasm') */
  backend?: 'wasm' | 'webgl' | 'webgpu' | 'webnn';
  /** WebNN provider options (only used when backend is 'webnn') */
  webnnOptions?: WebNNProviderOptions;
  /** Device type for WebNN/WebGPU (default: 'gpu' for high performance) */
  deviceType?: 'cpu' | 'gpu' | 'npu';
  /** Power preference for WebNN/WebGPU (default: 'high-performance') */
  powerPreference?: 'default' | 'low-power' | 'high-performance';
  /** Enable model caching (default: true) */
  cache?: boolean;
  /** Custom metadata for postprocessing */
  metadata?: any;
  /** Input normalization (default: { mean: [0, 0, 0], std: [1, 1, 1] }) */
  normalization?: {
    mean: number[];
    std: number[];
  };
  /** Input size for automatic preprocessing (optional) */
  inputSize?: [number, number];
  /** Keep aspect ratio during preprocessing (default: true) */
  keepAspectRatio?: boolean;
  /** Background color for letterbox (default: black) */
  backgroundColor?: string;
}

/**
 * Detection result with metadata
 */
export interface DetectionResult<T = any> {
  /** Raw model outputs */
  outputs: Record<string, ort.Tensor>;
  /** Processed results */
  data: T;
  /** Inference time in ms */
  inferenceTime: number;
  /** Input shape used */
  inputShape: number[];
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: Partial<CustomDetectorConfig> = {
  backend: 'webgpu',  // Default to WebGPU for better performance
  cache: true,
  keepAspectRatio: true,
  backgroundColor: '#000000',
  normalization: {
    mean: [0, 0, 0],
    std: [1, 1, 1],
  },
};

export class CustomDetector {
  private config: Required<CustomDetectorConfig>;
  private session: ort.InferenceSession | null = null;
  private initialized = false;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  constructor(config: CustomDetectorConfig) {
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
      outputNames: config.outputNames || [],
      inputShape: config.inputShape || [1, 3, 224, 224],
      normalization: config.normalization || { mean: [0, 0, 0], std: [1, 1, 1] },
    } as Required<CustomDetectorConfig>;
  }

  /**
   * Initialize the model
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    try {
      this.session = await loadOnnxSession({
        url: this.config.model,
        backend: this.config.backend,
        cache: this.config.cache,
        deviceType: this.config.deviceType,
        powerPreference: this.config.powerPreference,
        logPrefix: '[CustomDetector]',
      });

      // Auto-detect input/output names if not specified
      if (!this.config.inputName && this.session.inputNames.length > 0) {
        log.log(`Auto-detected input name: ${this.session.inputNames[0]}`);
      }

      if (this.config.outputNames.length === 0 && this.session.outputNames.length > 0) {
        this.config.outputNames = [...this.session.outputNames];
        log.log(`Auto-detected output names: ${this.config.outputNames}`);
      }

      log.log(`✅ Initialized (${this.config.backend})`);
      this.initialized = true;
    } catch (error) {
      log.error('❌ Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Run inference on canvas
   */
  async runFromCanvas<T = any>(canvas: HTMLCanvasElement): Promise<DetectionResult<T>> {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context from canvas');
    }

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return this.run<T>(imageData, canvas.width, canvas.height);
  }

  /**
   * Run inference on video
   */
  async runFromVideo<T = any>(
    video: HTMLVideoElement,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectionResult<T>> {
    if (video.readyState < 2) {
      throw new Error('Video not ready');
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

    return this.run<T>(imageData, canvas.width, canvas.height);
  }

  /**
   * Run inference on image
   */
  async runFromImage<T = any>(
    image: HTMLImageElement,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectionResult<T>> {
    if (!image.complete || !image.naturalWidth) {
      throw new Error('Image not loaded');
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

    return this.run<T>(imageData, canvas.width, canvas.height);
  }

  /**
   * Run inference on bitmap
   */
  async runFromBitmap<T = any>(
    bitmap: ImageBitmap,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectionResult<T>> {
    const canvas = targetCanvas || document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context from canvas');
    }

    ctx.drawImage(bitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    return this.run<T>(imageData, canvas.width, canvas.height);
  }

  /**
   * Run inference on file
   */
  async runFromFile<T = any>(
    file: File,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectionResult<T>> {
    const img = await loadImageFromFile(file);
    return this.runFromImage<T>(img, targetCanvas);
  }

  /**
   * Run inference on blob
   */
  async runFromBlob<T = any>(
    blob: Blob,
    targetCanvas?: HTMLCanvasElement
  ): Promise<DetectionResult<T>> {
    const bitmap = await loadBitmapFromBlob(blob);
    try {
      return await this.runFromBitmap<T>(bitmap, targetCanvas);
    } finally {
      bitmap.close();
    }
  }

  /**
   * Run inference with custom preprocessing
   */
  async run<T = any>(
    imageData: ImageData,
    width: number,
    height: number,
    metadata?: any
  ): Promise<DetectionResult<T>> {
    if (!this.initialized) {
      await this.init();
    }

    const startTime = performance.now();

    // Preprocess
    let inputTensor: ort.Tensor;

    if (this.config.preprocessing) {
      // Custom preprocessing
      const result = this.config.preprocessing(imageData, this.config);
      if (result instanceof Float32Array) {
        const [h, w] = this.config.inputSize || [height, width];
        inputTensor = new ort.Tensor('float32', result, [1, 3, h, w]);
      } else {
        inputTensor = result;
      }
    } else if (this.config.inputSize) {
      // Automatic preprocessing with letterbox
      inputTensor = this.preprocess(imageData, width, height, this.config.inputSize);
    } else {
      // Simple preprocessing - just normalize
      inputTensor = this.simplePreprocess(imageData);
    }

    // Get input name
    const inputName = this.config.inputName || this.session!.inputNames[0];

    // Run inference
    const feeds: Record<string, ort.Tensor> = {};
    feeds[inputName] = inputTensor;

    const results = await this.session!.run(feeds);

    // Postprocess
    let data: T;
    if (this.config.postprocessing) {
      data = this.config.postprocessing(results, metadata || this.config.metadata);
    } else {
      // Return raw outputs
      data = results as any;
    }

    const inferenceTime = performance.now() - startTime;

    return {
      outputs: results,
      data,
      inferenceTime,
      inputShape: [...inputTensor.dims],
    };
  }

  /**
   * Get model info
   */
  getModelInfo(): {
    inputNames: string[];
    outputNames: string[];
    inputCount: number;
    outputCount: number;
  } {
    if (!this.session) {
      throw new Error('Model not initialized. Call init() first.');
    }

    return {
      inputNames: [...this.session.inputNames],
      outputNames: [...this.session.outputNames],
      inputCount: this.session.inputNames.length,
      outputCount: this.session.outputNames.length,
    };
  }

  /**
   * Get tensor by name from outputs
   */
  getOutputTensor<T extends ort.Tensor = ort.Tensor>(
    outputs: Record<string, ort.Tensor>,
    name?: string
  ): T {
    const tensorName = name || this.config.outputNames[0] || this.session!.outputNames[0];
    return outputs[tensorName] as T;
  }

  /**
   * Simple preprocessing - just normalize to [0, 1] and convert to CHW
   */
  private simplePreprocess(imageData: ImageData): ort.Tensor {
    const { width, height, data } = imageData;
    const tensor = new Float32Array(3 * width * height);

    for (let i = 0; i < data.length; i += 4) {
      const pixelIdx = i / 4;
      tensor[pixelIdx] = data[i] / 255;
      tensor[pixelIdx + width * height] = data[i + 1] / 255;
      tensor[pixelIdx + 2 * width * height] = data[i + 2] / 255;
    }

    return new ort.Tensor('float32', tensor, [1, 3, height, width]);
  }

  /**
   * Preprocess with letterbox and normalization.
   *
   * `keepAspectRatio` (default true) preserves the source aspect ratio
   * and pads with `backgroundColor`; the browser-resampled drawImage
   * path is significantly faster than a manual pixel loop and is
   * faithful to the previous implementation.
   */
  private preprocess(
    imageData: ImageData,
    imgWidth: number,
    imgHeight: number,
    inputSize: [number, number]
  ): ort.Tensor {
    const [inputW, inputH] = inputSize;

    if (!this.canvas || !this.ctx) {
      this.canvas = document.createElement('canvas');
      this.canvas.width = inputW;
      this.canvas.height = inputH;
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true, alpha: false })!;
    }

    const ctx = this.ctx;

    // Stage source RGBA on a scratch canvas so we can use drawImage.
    const srcCanvas = document.createElement('canvas');
    const srcCtx = srcCanvas.getContext('2d')!;
    srcCanvas.width = imgWidth;
    srcCanvas.height = imgHeight;
    srcCtx.putImageData(imageData, 0, 0);

    if (this.config.keepAspectRatio) {
      const geom = letterboxGeometry(imgWidth, imgHeight, inputW, inputH);
      fillLetterbox(ctx, inputW, inputH, this.config.backgroundColor);
      ctx.drawImage(srcCanvas, 0, 0, imgWidth, imgHeight, geom.offX, geom.offY, geom.drawW, geom.drawH);
    } else {
      ctx.drawImage(srcCanvas, 0, 0, imgWidth, imgHeight, 0, 0, inputW, inputH);
    }

    const paddedData = ctx.getImageData(0, 0, inputW, inputH);
    const tensor = new Float32Array(inputW * inputH * 3);
    rgbaToCHW(paddedData.data, inputW, inputH, tensor);

    // Apply user-supplied normalization on top of the [0, 1] baseline.
    const { mean, std } = this.config.normalization;
    if (mean[0] !== 0 || mean[1] !== 0 || mean[2] !== 0 ||
        std[0] !== 1 || std[1] !== 1 || std[2] !== 1) {
      const plane = inputW * inputH;
      const inv0 = 1 / std[0], inv1 = 1 / std[1], inv2 = 1 / std[2];
      const meanScaled0 = mean[0] / 255, meanScaled1 = mean[1] / 255, meanScaled2 = mean[2] / 255;
      for (let i = 0; i < plane; i++) {
        tensor[i] = (tensor[i] - meanScaled0) * inv0;
        tensor[i + plane] = (tensor[i + plane] - meanScaled1) * inv1;
        tensor[i + 2 * plane] = (tensor[i + 2 * plane] - meanScaled2) * inv2;
      }
    }

    return new ort.Tensor('float32', tensor, [1, 3, inputH, inputW]);
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    if (this.session) {
      this.session.release();
      this.session = null;
    }
    this.initialized = false;
  }
}
