/**
 * rtmlib-ts - Real-Time Multi-Person Pose Estimation Library
 * TypeScript port of rtmlib for browser-based AI inference.
 *
 * 3D pose estimation is unified behind a single `Pose3DDetector` class that
 * composes two orthogonal selectors: `objectModel` (`'yolov8n' | 'yolov12n' |
 * 'yolo26n' | 'mediapipe'`) picks the person detector and `pose3dModel`
 * (`'rtmw3d' | 'instanthmr'`) picks the 3D pose model. All 8 combinations run
 * on ONNX Runtime Web; the InstantHMR pipeline additionally emits the
 * 70-keypoint MHR mesh + camera translation + body shape.
 */

// Solutions (High-level APIs)
export {
  ObjectDetector,
  COCO_CLASSES,
  type ObjectDetectorBackend,
  type ObjectDetectorConfig,
  type DetectedObject,
} from './solution/objectDetector';
export { PoseDetector, type PoseDetectorConfig } from './solution/poseDetector';
export {
  Pose3DDetector,
  type Pose3DObjectModel,
  type Pose3DModel,
  type Pose3DDetectorConfigGeneric,
  type Pose3DDetectorConfig,
  type Pose3DResult,
  type Pose3DStats,
  type Pose3DProfile,
  type Person3D,
  type Pose3DDetectorResult,
  type InstantHMR3DResult,
  type InstantHMRPerson,
  type InstantHMRKeypoint3D,
} from './solution/pose3dDetector';
export { CustomDetector, type CustomDetectorConfig, type DetectionResult } from './solution/customDetector';
export { AnimalDetector, ANIMAL_CLASSES, VITPOSE_MODELS, type VitPoseModelType, type DetectedAnimal, type AnimalKeypoint } from './solution/animalDetector';

// Models
export { YOLO12 } from './models/yolo12';
export { YOLO26 } from './models/yolo26';
// Default URLs for the three supported Ultralytics YOLO versions (the
// HuggingFace mirror that the rest of the demo fetches from). Useful when
// constructing a detector via `new ObjectDetector({ yoloVersion: 'yolo26n' })`
// — no need to look up the URL.
export { YOLO_VERSIONS, resolveYoloModelUrl, type YoloVersion } from './models/yoloModels';
// Default URL for the InstantHMR MHR-mesh checkpoint (used by
// Pose3DDetector's pose3dModel: 'instanthmr'). Re-exported here so
// consumers can pin the URL without pulling in models/instanthmr
// separately.
export { INSTANTHMR_MODEL_URL } from './models/instanthmr';

// MediaPipe Solutions (used internally by `Pose3DDetector` when
// `objectModel === 'mediapipe'`; also useful standalone.)
export { MediaPipeObjectDetector, type MediaPipeDetectedObject, type MediaPipeDetectionStats, type MediaPipeObjectDetectorConfig } from './solution/mediaPipeObjectDetector';
export { MediaPipePoseDetector, type MediaPipeLandmark, type MediaPipeWorldLandmark, type MediaPipePose, type MediaPipePoseStats, type MediaPipePoseDetectorConfig } from './solution/mediaPipePoseDetector';

// Visualization
export {
  drawBbox,
  drawSkeleton,
  drawDetectionsOnCanvas,
  drawPoseOnCanvas,
  drawResultsOnCanvas,
  drawMhr70OnCanvas,
  drawInfoPanel,
} from './visualization/draw';

// Model caching utilities
export {
  getCachedModel,
  isModelCached,
  preloadModels,
  clearModelCache,
  getCacheSize,
  getCacheInfo,
  EmptyModelResponseError,
} from './core/modelCache';

// Environment detection utilities
export {
  isBrowser,
  isSSR,
  getDocument,
  createCanvas,
} from './core/environment';

// ONNX Runtime Web initialization
export {
  initOnnxRuntimeWeb,
  getOnnxRuntime,
} from './core/onnxRuntime';

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
  mhr70,
  openpose18,
  openpose134,
} from './visualization/skeleton/index';

// Version
export const VERSION = '0.1.0';
