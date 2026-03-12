/**
 * Hand solution - hand pose estimation with 21 keypoints
 */

import { YOLOX } from '../models/yolox';
import { RTMPose } from '../models/rtmpose';
import { BBox } from '../types/index';

export class Hand {
  private detModel: YOLOX;
  private poseModel: RTMPose;

  constructor(
    det: string = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip',
    detInputSize: [number, number] = [320, 320],
    pose: string = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip',
    poseInputSize: [number, number] = [256, 256],
    toOpenpose: boolean = false,
    backend: 'onnxruntime' = 'onnxruntime',
    device: string = 'cpu'
  ) {
    this.detModel = new YOLOX(
      det,
      detInputSize,
      0.45,
      0.5,
      backend
    );

    this.poseModel = new RTMPose(
      pose,
      poseInputSize,
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
    imgHeight: number
  ): Promise<{ keypoints: number[][]; scores: number[] }> {
    const bboxes = await this.detModel.call(image, imgWidth, imgHeight);
    const result = await this.poseModel.call(image, imgWidth, imgHeight, bboxes);
    return result;
  }
}
