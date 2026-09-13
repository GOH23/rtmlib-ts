/**
 * Basic types for rtmlib-ts
 * Based on rtmlib Python library
 */

export interface Keypoint {
  x: number;
  y: number;
  score: number;
  id: number;
}

export interface BodyResult {
  keypoints: Array<Keypoint | null>;
  totalScore: number;
  totalParts: number;
}

export type HandResult = Keypoint[];
export type FaceResult = Keypoint[];

export interface PoseResult {
  body: BodyResult;
  leftHand: HandResult | null;
  rightHand: HandResult | null;
  face: FaceResult | null;
}

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  bbox: BBox;
  score: number;
  classId: number;
}

export interface ModelConfig {
  det: string;
  detInputSize: [number, number];
  pose: string;
  poseInputSize: [number, number];
}

export type ModeType = 'performance' | 'lightweight' | 'balanced';

/**
 * ONNX Runtime Web execution providers supported by `rtmlib-ts`.
 *
 * (The original Python `rtmlib` library also accepted `'opencv'`,
 * `'onnxruntime'`, and `'openvino'`; those legacy strings are not
 * meaningful in the browser runtime and have been removed.)
 */
export type BackendType = 'wasm' | 'webgl' | 'webgpu' | 'webnn';

/**
 * Device hint for WebNN / WebGPU. `npu` is only meaningful with WebNN.
 * `'mps'` (Apple Metal) was inherited from the Python source but is
 * not reachable from a browser.
 */
export type DeviceType = 'cpu' | 'gpu' | 'npu';

export interface ImageData {
  data: Uint8Array;
  width: number;
  height: number;
  channels: number;
}

export type RGBImage = ImageData;
export type BGRImage = ImageData;

export interface WebNNProviderOptions {
  name: 'webnn';
  deviceType?: 'cpu' | 'gpu' | 'npu';
  powerPreference?: 'default' | 'low-power' | 'high-performance';
}

export type WebNNProviderOptionsOrUndefined = WebNNProviderOptions | undefined;
