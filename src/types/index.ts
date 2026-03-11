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

export type BackendType = 'opencv' | 'onnxruntime' | 'openvino';
export type DeviceType = 'cpu' | 'cuda' | 'mps' | string;

export interface ImageData {
  data: Uint8Array;
  width: number;
  height: number;
  channels: number;
}

export type RGBImage = ImageData;
export type BGRImage = ImageData;
