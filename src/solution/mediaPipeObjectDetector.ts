/**
 * MediaPipeObjectDetector - Object Detection using MediaPipe Tasks Vision
 * Provides multi-class object detection using MediaPipe's efficient models
 *
 * @example
 * ```typescript
 * // Initialize with default model
 * const detector = new MediaPipeObjectDetector({
 *   scoreThreshold: 0.5,
 * });
 * await detector.init();
 *
 * // Detect from canvas
 * const objects = await detector.detectFromCanvas(canvas);
 *
 * // Filter by specific classes
 * detector.setCategoryAllowlist(['person', 'car', 'dog']);
 * ```
 */

import { FilesetResolver, ObjectDetector as MPObjectDetector } from '@mediapipe/tasks-vision';
import type { ObjectDetector as MPObjectDetectorType } from '@mediapipe/tasks-vision';
import { loadMediaPipeModelWithCache } from '../core/mediaPipeCache';

/**
 * Detected object with bounding box and class
 */
export interface MediaPipeDetectedObject {
  /** Bounding box coordinates */
  bbox: {
    x1: number;
    y1: number;
    width: number;
    height: number;
  };
  /** Category name (e.g., "person", "car") */
  categoryName: string;
  /** Category index */
  categoryId: number;
  /** Detection confidence (0-1) */
  score: number;
  /** Localized display name (if available) */
  displayName?: string;
}

/**
 * Detection statistics
 */
export interface MediaPipeDetectionStats {
  /** Total number of detections */
  totalCount: number;
  /** Detections per class */
  classCounts: Record<string, number>;
  /** Inference time (ms) */
  inferenceTime: number;
}

/**
 * Configuration options for MediaPipeObjectDetector
 */
export interface MediaPipeObjectDetectorConfig {
  /** Path to .task model file (optional - uses default EfficientDet Lite0 if not specified) */
  modelPath?: string;
  /** Confidence threshold for filtering results (default: 0.5) */
  scoreThreshold?: number;
  /** Maximum number of detections (-1 for all, default: -1) */
  maxResults?: number;
  /** Only detect these categories (optional) */
  categoryAllowlist?: string[];
  /** Exclude these categories (optional, mutually exclusive with allowlist) */
  categoryDenylist?: string[];
  /** Locale for display names (default: 'en') */
  displayNamesLocale?: string;
  /** Running mode: IMAGE for single images, VIDEO for video streams */
  runningMode?: 'IMAGE' | 'VIDEO';
  /** Enable model caching (default: true) */
  cache?: boolean;
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: Required<MediaPipeObjectDetectorConfig> = {
  modelPath: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite',
  scoreThreshold: 0.5,
  maxResults: -1,
  categoryAllowlist: [],
  categoryDenylist: [],
  displayNamesLocale: 'en',
  runningMode: 'IMAGE',
  cache: true,
};

export class MediaPipeObjectDetector {
  private config: Required<MediaPipeObjectDetectorConfig>;
  private detector: MPObjectDetector | null = null;
  private initialized = false;
  private vision: any = null;

  constructor(config: MediaPipeObjectDetectorConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Initialize the MediaPipe Object Detector
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    try {
      // Initialize FilesetResolver for Vision Tasks
      this.vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );

      // Load model with caching (if enabled)
      const startTime = performance.now();
      let modelBuffer: ArrayBuffer;
      
      if (this.config.cache) {
        modelBuffer = await loadMediaPipeModelWithCache(this.config.modelPath);
      } else {
        const response = await fetch(this.config.modelPath);
        if (!response.ok) {
          throw new Error(`Failed to fetch model: HTTP ${response.status}`);
        }
        modelBuffer = await response.arrayBuffer();
      }
      
      const loadTime = Math.round(performance.now() - startTime);

      // Create Object Detector using baseOptions with modelAssetBuffer
      const createStart = performance.now();
      
      this.detector = await MPObjectDetector.createFromOptions(this.vision, {
        baseOptions: {
          modelAssetBuffer: new Uint8Array(modelBuffer),
        },
        scoreThreshold: this.config.scoreThreshold,
        maxResults: this.config.maxResults,
        categoryAllowlist: this.config.categoryAllowlist.length > 0 ? this.config.categoryAllowlist : undefined,
        categoryDenylist: this.config.categoryDenylist.length > 0 ? this.config.categoryDenylist : undefined,
        runningMode: this.config.runningMode,
      });

      const createTime = Math.round(performance.now() - createStart);

      this.initialized = true;
    } catch (error) {
      console.error('[MediaPipeObjectDetector] Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Set which categories to detect (allowlist)
   * @param categories - Array of category names to allow
   */
  setCategoryAllowlist(categories: string[]): void {
    this.config.categoryAllowlist = categories;
    this.config.categoryDenylist = [];
    this.updateOptions();
  }

  /**
   * Set categories to exclude (denylist)
   * @param categories - Array of category names to deny
   */
  setCategoryDenylist(categories: string[]): void {
    this.config.categoryDenylist = categories;
    this.config.categoryAllowlist = [];
    this.updateOptions();
  }

  /**
   * Clear category filters (detect all classes)
   */
  clearCategoryFilter(): void {
    this.config.categoryAllowlist = [];
    this.config.categoryDenylist = [];
    this.updateOptions();
  }

  /**
   * Update detector options
   */
  private async updateOptions(): Promise<void> {
    if (!this.detector) return;

    await this.detector.setOptions({
      scoreThreshold: this.config.scoreThreshold,
      maxResults: this.config.maxResults,
      categoryAllowlist: this.config.categoryAllowlist.length > 0 ? this.config.categoryAllowlist : undefined,
      categoryDenylist: this.config.categoryDenylist.length > 0 ? this.config.categoryDenylist : undefined,
    });
  }

  /**
   * Detect objects from HTMLCanvasElement
   */
  async detectFromCanvas(canvas: HTMLCanvasElement): Promise<MediaPipeDetectedObject[]> {
    if (!this.initialized) {
      await this.init();
    }

    const startTime = performance.now();
    const result = this.detector!.detect(canvas);
    const inferenceTime = performance.now() - startTime;

    const detections = this.convertDetections(result);
    (detections as any).stats = this.calculateStats(detections, inferenceTime);

    return detections;
  }

  /**
   * Detect objects from HTMLVideoElement
   */
  async detectFromVideo(video: HTMLVideoElement): Promise<MediaPipeDetectedObject[]> {
    if (!this.initialized) {
      await this.init();
    }

    // Switch to VIDEO mode if not already
    if (this.config.runningMode !== 'VIDEO') {
      await this.detector!.setOptions({ runningMode: 'VIDEO' });
      this.config.runningMode = 'VIDEO';
    }

    const startTime = performance.now();
    const result = this.detector!.detectForVideo(video, performance.now());
    const inferenceTime = performance.now() - startTime;

    const detections = this.convertDetections(result);
    (detections as any).stats = this.calculateStats(detections, inferenceTime);

    return detections;
  }

  /**
   * Detect objects from HTMLImageElement
   */
  async detectFromImage(image: HTMLImageElement): Promise<MediaPipeDetectedObject[]> {
    if (!this.initialized) {
      await this.init();
    }

    const startTime = performance.now();
    const result = this.detector!.detect(image);
    const inferenceTime = performance.now() - startTime;

    const detections = this.convertDetections(result);
    (detections as any).stats = this.calculateStats(detections, inferenceTime);

    return detections;
  }

  /**
   * Detect objects from ImageBitmap
   */
  async detectFromBitmap(bitmap: ImageBitmap): Promise<MediaPipeDetectedObject[]> {
    if (!this.initialized) {
      await this.init();
    }

    // Create temporary canvas
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(bitmap, 0, 0);

    return this.detectFromCanvas(canvas);
  }

  /**
   * Detect objects from File
   */
  async detectFromFile(file: File): Promise<MediaPipeDetectedObject[]> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = async () => {
        try {
          const results = await this.detectFromImage(img);
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
   * Detect objects from Blob
   */
  async detectFromBlob(blob: Blob): Promise<MediaPipeDetectedObject[]> {
    const bitmap = await createImageBitmap(blob);
    const results = await this.detectFromBitmap(bitmap);
    bitmap.close();
    return results;
  }

  /**
   * Convert MediaPipe detections to our format
   */
  private convertDetections(result: any): MediaPipeDetectedObject[] {
    const detections: MediaPipeDetectedObject[] = [];

    if (!result.detections) return detections;

    for (const detection of result.detections) {
      const bbox = detection.boundingBox;
      const category = detection.categories[0];

      detections.push({
        bbox: {
          x1: bbox.originX,
          y1: bbox.originY,
          width: bbox.width,
          height: bbox.height,
        },
        categoryName: category.categoryName,
        categoryId: category.index,
        score: category.score,
        displayName: category.displayName,
      });
    }

    return detections;
  }

  /**
   * Calculate detection statistics
   */
  private calculateStats(
    detections: MediaPipeDetectedObject[],
    inferenceTime: number
  ): MediaPipeDetectionStats {
    const classCounts: Record<string, number> = {};

    detections.forEach((det) => {
      classCounts[det.categoryName] = (classCounts[det.categoryName] || 0) + 1;
    });

    return {
      totalCount: detections.length,
      classCounts,
      inferenceTime: Math.round(inferenceTime),
    };
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    if (this.detector) {
      this.detector.close();
      this.detector = null;
    }
    this.initialized = false;
  }
}
