/**
 * MediaPipePoseDetector - Pose Landmark Detection using MediaPipe Tasks Vision
 * Provides human pose estimation with 33 keypoints using MediaPipe's efficient models
 *
 * @example
 * ```typescript
 * // Initialize with default model
 * const detector = new MediaPipePoseDetector({
 *   minPoseDetectionConfidence: 0.5,
 *   numPoses: 3,
 * });
 * await detector.init();
 *
 * // Detect poses from canvas
 * const poses = await detector.detectFromCanvas(canvas);
 *
 * // Access landmarks
 * poses.forEach(pose => {
 *   console.log(`Pose confidence: ${pose.score}`);
 *   pose.landmarks.forEach(landmark => {
 *     console.log(`  ${landmark.name}: (${landmark.x.toFixed(3)}, ${landmark.y.toFixed(3)}, ${landmark.z.toFixed(3)})`);
 *   });
 * });
 * ```
 */

import { FilesetResolver, PoseLandmarker as MediaPipePoseLandmarker } from '@mediapipe/tasks-vision';

/**
 * Single landmark with 3D coordinates
 */
export interface MediaPipeLandmark {
  /** X coordinate (normalized [0, 1] relative to image width) */
  x: number;
  /** Y coordinate (normalized [0, 1] relative to image height) */
  y: number;
  /** Z coordinate (depth in meters, smaller = closer to camera) */
  z: number;
  /** Visibility score (0-1) - likelihood that landmark is visible */
  visibility: number;
  /** Presence score (0-1) - likelihood that landmark is present */
  presence: number;
  /** Landmark name (e.g., "nose", "left_shoulder") */
  name: string;
  /** Landmark index (0-32) */
  index: number;
}

/**
 * World coordinates for a landmark (in meters)
 */
export interface MediaPipeWorldLandmark extends MediaPipeLandmark {
  /** X coordinate in meters (origin: mid-hips) */
  worldX: number;
  /** Y coordinate in meters (origin: mid-hips) */
  worldY: number;
  /** Z coordinate in meters (origin: mid-hips) */
  worldZ: number;
}

/**
 * Detected person with pose landmarks
 */
export interface MediaPipePose {
  /** Bounding box around the person */
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
  /** 33 normalized landmarks */
  landmarks: MediaPipeLandmark[];
  /** 33 world landmarks (in meters) */
  worldLandmarks: MediaPipeWorldLandmark[];
  /** Overall pose confidence score */
  score: number;
}

/**
 * Detection statistics
 */
export interface MediaPipePoseStats {
  /** Number of detected poses */
  poseCount: number;
  /** Inference time (ms) */
  inferenceTime: number;
}

/**
 * Configuration options for MediaPipePoseDetector
 */
export interface MediaPipePoseDetectorConfig {
  /** Path to .task model file (optional - uses default if not specified) */
  modelPath?: string;
  /** Maximum number of poses to detect (default: 3) */
  numPoses?: number;
  /** Minimum confidence for pose detection (default: 0.5) */
  minPoseDetectionConfidence?: number;
  /** Minimum confidence for pose presence (default: 0.5) */
  minPosePresenceConfidence?: number;
  /** Minimum confidence for pose tracking (default: 0.5) */
  minTrackingConfidence?: number;
  /** Output segmentation masks (default: false) */
  outputSegmentationMasks?: boolean;
  /** Running mode: IMAGE for single images, VIDEO for video streams */
  runningMode?: 'IMAGE' | 'VIDEO';
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: Required<MediaPipePoseDetectorConfig> = {
  modelPath: 'https://storage.googleapis.com/mediapipe-tasks/pose_landmarker/pose_landmarker_full.task',
  numPoses: 3,
  minPoseDetectionConfidence: 0.5,
  minPosePresenceConfidence: 0.5,
  minTrackingConfidence: 0.5,
  outputSegmentationMasks: false,
  runningMode: 'IMAGE',
};

/**
 * COCO-style 33 keypoint names (MediaPipe BlazePose)
 */
const KEYPOINT_NAMES = [
  'nose',           // 0
  'left_eye',       // 1
  'right_eye',      // 2
  'left_ear',       // 3
  'right_ear',      // 4
  'left_shoulder',  // 5
  'right_shoulder', // 6
  'left_elbow',     // 7
  'right_elbow',    // 8
  'left_wrist',     // 9
  'right_wrist',    // 10
  'left_hip',       // 11
  'right_hip',      // 12
  'left_knee',      // 13
  'right_knee',     // 14
  'left_ankle',     // 15
  'right_ankle',    // 16
  'left_heel',      // 17
  'right_heel',     // 18
  'left_foot_index',// 19
  'right_foot_index',// 20
  // Additional face landmarks (21-32)
  'left_eye_inner', // 21
  'left_eye',       // 22 (center)
  'left_eye_outer', // 23
  'right_eye_inner',// 24
  'right_eye',      // 25 (center)
  'right_eye_outer',// 26
  'nose_tip',       // 27
  'mouth_left',     // 28
  'mouth_right',    // 29
  'left_pupil',     // 30
  'right_pupil',    // 31
  'left_ear_tragion',// 32
  'right_ear_tragion',// 33
];

// Note: MediaPipe returns 33 landmarks, but we'll use the first 17 for COCO compatibility
const COCO_KEYPOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

export class MediaPipePoseDetector {
  private config: Required<MediaPipePoseDetectorConfig>;
  private detector: MediaPipePoseLandmarker | null = null;
  private initialized = false;
  private vision: any = null;

  constructor(config: MediaPipePoseDetectorConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Initialize the MediaPipe Pose Landmarker
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    try {
      console.log('[MediaPipePoseDetector] Initializing...');

      // Initialize FilesetResolver for Vision Tasks
      this.vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );

      // Create Pose Landmarker
      this.detector = await MediaPipePoseLandmarker.createFromOptions(this.vision, {
        baseOptions: {
          modelAssetPath: this.config.modelPath,
        },
        numPoses: this.config.numPoses,
        minPoseDetectionConfidence: this.config.minPoseDetectionConfidence,
        minPosePresenceConfidence: this.config.minPosePresenceConfidence,
        minTrackingConfidence: this.config.minTrackingConfidence,
        outputSegmentationMasks: this.config.outputSegmentationMasks,
        runningMode: this.config.runningMode,
      });

      this.initialized = true;
      console.log('[MediaPipePoseDetector] ✅ Initialized');
    } catch (error) {
      console.error('[MediaPipePoseDetector] ❌ Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Detect poses from HTMLCanvasElement
   */
  async detectFromCanvas(canvas: HTMLCanvasElement): Promise<MediaPipePose[]> {
    if (!this.initialized) {
      await this.init();
    }

    const startTime = performance.now();
    const result = this.detector!.detect(canvas);
    const inferenceTime = performance.now() - startTime;

    const poses = this.convertPoses(result, canvas.width, canvas.height);
    (poses as any).stats = {
      poseCount: poses.length,
      inferenceTime: Math.round(inferenceTime),
    } as MediaPipePoseStats;

    return poses;
  }

  /**
   * Detect poses from HTMLVideoElement
   */
  async detectFromVideo(video: HTMLVideoElement): Promise<MediaPipePose[]> {
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

    const poses = this.convertPoses(result, video.videoWidth, video.videoHeight);
    (poses as any).stats = {
      poseCount: poses.length,
      inferenceTime: Math.round(inferenceTime),
    } as MediaPipePoseStats;

    return poses;
  }

  /**
   * Detect poses from HTMLImageElement
   */
  async detectFromImage(image: HTMLImageElement): Promise<MediaPipePose[]> {
    if (!this.initialized) {
      await this.init();
    }

    const startTime = performance.now();
    const result = this.detector!.detect(image);
    const inferenceTime = performance.now() - startTime;

    const poses = this.convertPoses(result, image.naturalWidth, image.naturalHeight);
    (poses as any).stats = {
      poseCount: poses.length,
      inferenceTime: Math.round(inferenceTime),
    } as MediaPipePoseStats;

    return poses;
  }

  /**
   * Detect poses from ImageBitmap
   */
  async detectFromBitmap(bitmap: ImageBitmap): Promise<MediaPipePose[]> {
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
   * Detect poses from File
   */
  async detectFromFile(file: File): Promise<MediaPipePose[]> {
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
   * Detect poses from Blob
   */
  async detectFromBlob(blob: Blob): Promise<MediaPipePose[]> {
    const bitmap = await createImageBitmap(blob);
    const results = await this.detectFromBitmap(bitmap);
    bitmap.close();
    return results;
  }

  /**
   * Convert MediaPipe pose results to our format
   */
  private convertPoses(result: any, width: number, height: number): MediaPipePose[] {
    const poses: MediaPipePose[] = [];

    if (!result.landmarks) return poses;

    for (let i = 0; i < result.landmarks.length; i++) {
      const landmarks = result.landmarks[i];
      const worldLandmarks = result.worldLandmarks?.[i] || [];

      // Calculate bounding box from landmarks
      let minX = Infinity, minY = Infinity;
      let maxX = -Infinity, maxY = -Infinity;

      for (const lm of landmarks) {
        const px = lm.x * width;
        const py = lm.y * height;
        minX = Math.min(minX, px);
        minY = Math.min(minY, py);
        maxX = Math.max(maxX, px);
        maxY = Math.max(maxY, py);
      }

      // Calculate average confidence
      const avgVisibility = landmarks.reduce((sum: number, lm: any) => sum + (lm.visibility || 0), 0) / landmarks.length;

      // Convert landmarks
      const convertedLandmarks: MediaPipeLandmark[] = landmarks.map((lm: any, idx: number) => ({
        x: lm.x,
        y: lm.y,
        z: lm.z || 0,
        visibility: lm.visibility || 0,
        presence: lm.presence || 0,
        name: KEYPOINT_NAMES[idx] || `landmark_${idx}`,
        index: idx,
      }));

      // Convert world landmarks
      const convertedWorldLandmarks: MediaPipeWorldLandmark[] = worldLandmarks.map((lm: any, idx: number) => ({
        x: lm.x || 0,
        y: lm.y || 0,
        z: lm.z || 0,
        worldX: lm.x || 0,
        worldY: lm.y || 0,
        worldZ: lm.z || 0,
        visibility: lm.visibility || 0,
        presence: lm.presence || 0,
        name: KEYPOINT_NAMES[idx] || `landmark_${idx}`,
        index: idx,
      }));

      poses.push({
        bbox: {
          x1: Math.max(0, minX),
          y1: Math.max(0, minY),
          x2: Math.min(width, maxX),
          y2: Math.min(height, maxY),
        },
        landmarks: convertedLandmarks,
        worldLandmarks: convertedWorldLandmarks,
        score: avgVisibility,
      });
    }

    return poses;
  }

  /**
   * Get COCO-compatible keypoints (17 points)
   */
  getCocoKeypoints(pose: MediaPipePose): MediaPipeLandmark[] {
    return COCO_KEYPOINT_INDICES.map(idx => pose.landmarks[idx]);
  }

  /**
   * Update detector options
   */
  async updateOptions(options: Partial<MediaPipePoseDetectorConfig>): Promise<void> {
    if (!this.detector) return;

    const newOptions: any = {};
    if (options.numPoses !== undefined) {
      newOptions.numPoses = options.numPoses;
      this.config.numPoses = options.numPoses;
    }
    if (options.minPoseDetectionConfidence !== undefined) {
      newOptions.minPoseDetectionConfidence = options.minPoseDetectionConfidence;
      this.config.minPoseDetectionConfidence = options.minPoseDetectionConfidence;
    }
    if (options.minPosePresenceConfidence !== undefined) {
      newOptions.minPosePresenceConfidence = options.minPosePresenceConfidence;
      this.config.minPosePresenceConfidence = options.minPosePresenceConfidence;
    }
    if (options.minTrackingConfidence !== undefined) {
      newOptions.minTrackingConfidence = options.minTrackingConfidence;
      this.config.minTrackingConfidence = options.minTrackingConfidence;
    }
    if (options.runningMode !== undefined) {
      newOptions.runningMode = options.runningMode;
      this.config.runningMode = options.runningMode;
    }

    await this.detector.setOptions(newOptions);
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
