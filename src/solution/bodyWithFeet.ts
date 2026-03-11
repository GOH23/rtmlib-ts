/**
 * BodyWithFeet solution - body pose estimation with 26 keypoints (including feet)
 */

import { YOLOX } from '../models/yolox';
import { RTMPose } from '../models/rtmpose';
import { ModeType, ModelConfig } from '../types/index';

export class BodyWithFeet {
  private detModel: YOLOX;
  private poseModel: RTMPose;

  private static readonly MODE: Record<ModeType, ModelConfig> = {
    performance: {
      det: 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_l_8xb8-300e_humanart-ce1d7a62.zip',
      detInputSize: [640, 640],
      pose: 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-384x288-734182ce_20230605.zip',
      poseInputSize: [288, 384],
    },
    lightweight: {
      det: 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip',
      detInputSize: [416, 416],
      pose: 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.zip',
      poseInputSize: [192, 256],
    },
    balanced: {
      det: 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
      detInputSize: [640, 640],
      pose: 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip',
      poseInputSize: [192, 256],
    },
  };

  constructor(
    det: string | null = null,
    detInputSize: [number, number] = [640, 640],
    pose: string | null = null,
    poseInputSize: [number, number] = [288, 384],
    mode: ModeType = 'balanced',
    toOpenpose: boolean = false,
    backend: 'onnxruntime' = 'onnxruntime',
    device: string = 'cpu'
  ) {
    let finalDet = det;
    let finalDetInputSize = detInputSize;
    let finalPose = pose;
    let finalPoseInputSize = poseInputSize;

    if (det === null) {
      finalDet = BodyWithFeet.MODE[mode].det;
      finalDetInputSize = BodyWithFeet.MODE[mode].detInputSize;
    }

    if (pose === null) {
      finalPose = BodyWithFeet.MODE[mode].pose;
      finalPoseInputSize = BodyWithFeet.MODE[mode].poseInputSize;
    }

    this.detModel = new YOLOX(
      finalDet!,
      finalDetInputSize,
      0.45,
      0.7,
      backend,
      device
    );

    this.poseModel = new RTMPose(
      finalPose!,
      finalPoseInputSize,
      toOpenpose,
      backend,
      device
    );
  }

  async init(): Promise<void> {
    await this.detModel.init();
    await this.poseModel.init();
  }

  async call(
    image: Uint8Array,
    imgWidth: number,
    imgHeight: number
  ): Promise<{ keypoints: number[][]; scores: number[] }> {
    const bboxes = await this.detModel.call(image, imgWidth, imgHeight);
    const result = await this.poseModel.call(image, imgWidth, imgHeight, bboxes);
    return result;
  }
}
