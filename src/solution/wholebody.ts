/**
 * Wholebody solution - combines detection and pose estimation
 * Based on rtmlib Wholebody class
 * Supports YOLO12 and YOLOX detectors with RTMW pose model
 */

import { YOLOX } from '../models/yolox';
import { YOLO12 } from '../models/yolo12';
import { RTMPose } from '../models/rtmpose';
import { BBox, ModeType, ModelConfig } from '../types';

export class Wholebody {
  private detModel: YOLOX | YOLO12;
  private poseModel: RTMPose;
  private detectorType: 'yolox' | 'yolo12';

  private static readonly MODE: Record<ModeType, ModelConfig> = {
    performance: {
      det: 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
      detInputSize: [640, 640],
      pose: 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip',
      poseInputSize: [384, 288],
    },
    lightweight: {
      det: 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip',
      detInputSize: [416, 416],
      pose: 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip',
      poseInputSize: [256, 192],
    },
    balanced: {
      det: 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
      detInputSize: [640, 640],
      pose: 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip',
      poseInputSize: [256, 192],
    },
  };

  constructor(
    det: string | null = null,
    detInputSize: [number, number] = [640, 640],
    pose: string | null = null,
    poseInputSize: [number, number] = [384, 288],
    mode: ModeType = 'balanced',
    toOpenpose: boolean = false,
    backend: 'onnxruntime' = 'onnxruntime',
    device: string = 'cpu',
    detectorType: 'yolox' | 'yolo12' = 'yolox'
  ) {
    this.detectorType = detectorType;

    // Use mode config if det/pose not specified
    let finalDet = det;
    let finalDetInputSize = detInputSize;
    let finalPose = pose;
    let finalPoseInputSize = poseInputSize;

    if (det === null) {
      finalDet = Wholebody.MODE[mode].det;
      finalDetInputSize = Wholebody.MODE[mode].detInputSize;
    }

    if (pose === null) {
      finalPose = Wholebody.MODE[mode].pose;
      finalPoseInputSize = Wholebody.MODE[mode].poseInputSize;
    }

    // Initialize detector based on type
    if (detectorType === 'yolo12') {
      this.detModel = new YOLO12(
        finalDet!,
        finalDetInputSize,
        0.45,
        0.5,
        backend
      );
    } else {
      this.detModel = new YOLOX(
        finalDet!,
        finalDetInputSize,
        0.45,
        0.7,
        backend
      );
    }

    this.poseModel = new RTMPose(
      finalPose!,
      finalPoseInputSize,
      toOpenpose,
      backend
    );
  }

  async init(): Promise<void> {
    await this.detModel.init();
    await this.poseModel.init();
  }

  async call(
    image: Uint8Array,
    imgWidth: number,
    imgHeight: number,
    bboxes?: BBox[]
  ): Promise<{ keypoints: number[][]; scores: number[] }> {
    // Run detection if bboxes not provided
    let finalBboxes = bboxes;
    if (!finalBboxes || finalBboxes.length === 0) {
      const detections: any[] = await this.detModel.call(image, imgWidth, imgHeight);
      // Convert Detection[] or BBox[] to BBox[]
      finalBboxes = detections.map((d: any) => {
        if ('bbox' in d) {
          // Detection type
          return {
            x1: d.bbox.x1,
            y1: d.bbox.y1,
            x2: d.bbox.x2,
            y2: d.bbox.y2,
          };
        }
        // Already BBox type
        return d;
      });
    }

    // Run pose estimation
    const result = await this.poseModel.call(image, imgWidth, imgHeight, finalBboxes);

    return result;
  }
}
