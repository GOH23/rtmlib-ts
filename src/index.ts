/**
 * rtmlib-ts - Real-Time Multi-Person Pose Estimation Library
 *
 * TypeScript port of rtmlib Python library
 * Based on RTMPose, DWPose, RTMO, RTMW models
 */

// Models
export { YOLOX } from './models/yolox';
export { YOLO12 } from './models/yolo12';
export { RTMPose } from './models/rtmpose';

// Solutions (High-level APIs)
export { ObjectDetector, COCO_CLASSES } from './solution/objectDetector';
export { PoseDetector } from './solution/poseDetector';
export { Wholebody } from './solution/wholebody';
export { Body } from './solution/body';
export { Hand } from './solution/hand';
export { BodyWithFeet } from './solution/bodyWithFeet';
export { PoseTracker } from './solution/poseTracker';

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
