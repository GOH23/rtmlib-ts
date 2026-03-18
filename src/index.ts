/**
 * rtmlib-ts - Real-Time Multi-Person Pose Estimation Library
 * TypeScript port of rtmlib for browser-based AI inference
 */

// Solutions (High-level APIs)
export { ObjectDetector, COCO_CLASSES } from './solution/objectDetector';
export { PoseDetector } from './solution/poseDetector';
export { Pose3DDetector, type Person3D, type Pose3DStats } from './solution/pose3dDetector';
export { CustomDetector, type CustomDetectorConfig, type DetectionResult } from './solution/customDetector';
export { AnimalDetector, ANIMAL_CLASSES, VITPOSE_MODELS, type VitPoseModelType, type DetectedAnimal, type AnimalKeypoint } from './solution/animalDetector';

// MediaPipe Solutions
export { MediaPipeObjectDetector, type MediaPipeDetectedObject, type MediaPipeDetectionStats, type MediaPipeObjectDetectorConfig } from './solution/mediaPipeObjectDetector';
export { MediaPipePoseDetector, type MediaPipeLandmark, type MediaPipeWorldLandmark, type MediaPipePose, type MediaPipePoseStats, type MediaPipePoseDetectorConfig } from './solution/mediaPipePoseDetector';
export { MediaPipeObject3DPoseDetector, type MediaPipeObject3DPoseDetectorConfig, type Person3DWithBBox } from './solution/mediaPipeObject3DPoseDetector';

// Visualization
export { drawBbox, drawSkeleton, drawDetectionsOnCanvas, drawPoseOnCanvas, drawResultsOnCanvas } from './visualization/draw';

// Model caching utilities
export {
  getCachedModel,
  isModelCached,
  preloadModels,
  clearModelCache,
  getCacheSize,
  getCacheInfo,
} from './core/modelCache';

// MediaPipe cache utilities
export {
  cacheMediaPipeModel,
  getCachedMediaPipeModel,
  isMediaPipeModelCached,
  clearMediaPipeCache,
  getMediaPipeCacheInfo,
  loadMediaPipeModelWithCache,
} from './core/mediaPipeCache';

// Types
export type {
  Keypoint,
  BodyResult,
  HandResult,
  FaceResult,
  PoseResult,
  BBox,
  Detection,
  ModelConfig,
  ModeType,
  BackendType,
  DeviceType,
  ImageData,
  RGBImage,
  BGRImage,
  WebNNProviderOptions,
} from './types/index';

// Skeleton configurations
export {
  coco17,
  coco133,
  hand21,
  halpe26,
  openpose18,
  openpose134,
} from './visualization/skeleton/index';

// Version
export const VERSION = '0.0.1';
