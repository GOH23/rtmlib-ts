/**
 * Wholebody3D solution - combines detection and 3D pose estimation
 * Based on rtmlib Wholebody3d class
 * Uses YOLOX detector with RTMW3D pose model
 */

import { YOLOX } from '../models/yolox';
import { RTMPose3D } from '../models/rtmpose3d';
import { BBox, ModeType, ModelConfig, BackendType } from '../types';

export interface Wholebody3DResult {
  keypoints: number[][][];
  scores: number[][];
  keypointsSimcc: number[][][];
  keypoints2d: number[][][];
}

export class Wholebody3D {
  private detModel: YOLOX;
  private poseModel: RTMPose3D;

  private static readonly MODE: Record<ModeType, ModelConfig> = {
    performance: {
      det: 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
      detInputSize: [640, 640],
      pose: 'https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx',
      poseInputSize: [288, 384],  // [width=288, height=384] - creates tensor [1,3,384,288]
    },
    lightweight: {
      det: 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip',
      detInputSize: [416, 416],
      pose: 'https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx',
      poseInputSize: [192, 256],  // [width=192, height=256]
    },
    balanced: {
      det: 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
      detInputSize: [640, 640],
      pose: 'https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx',
      poseInputSize: [288, 384],  // [width=288, height=384] - creates tensor [1,3,384,288]
    },
  };

  constructor(
    det: string | null = null,
    detInputSize: [number, number] = [640, 640],
    pose: string | null = null,
    poseInputSize: [number, number] = [288, 384],  // [width=288, height=384]
    mode: ModeType = 'balanced',
    toOpenpose: boolean = false,
    backend: BackendType = 'webgpu'
  ) {
    // Use mode config if det/pose not specified
    let finalDet = det;
    let finalDetInputSize = detInputSize;
    let finalPose = pose;
    let finalPoseInputSize = poseInputSize;

    if (det === null) {
      finalDet = Wholebody3D.MODE[mode].det;
      finalDetInputSize = Wholebody3D.MODE[mode].detInputSize;
    }

    if (pose === null) {
      finalPose = Wholebody3D.MODE[mode].pose;
      finalPoseInputSize = Wholebody3D.MODE[mode].poseInputSize;
    }

    this.detModel = new YOLOX(
      finalDet!,
      finalDetInputSize,
      0.45,
      0.7,
      backend
    );

    this.poseModel = new RTMPose3D(
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
  ): Promise<Wholebody3DResult> {
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

    // Run 3D pose estimation
    const result = await this.poseModel.call(
      image,
      imgWidth,
      imgHeight,
      finalBboxes
    );

    return result;
  }
}
