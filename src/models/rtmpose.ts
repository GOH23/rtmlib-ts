/**
 * RTMPose model for pose estimation
 * Supports RTMPose, DWPose, RTMW variants
 * Uses onnxruntime-web for browser compatibility
 */

import { BaseTool } from '../core/base';
import { BBox } from '../types/index';

export class RTMPose extends BaseTool {
  private toOpenpose: boolean;
  private simccSplitRatio: number = 2.0;
  private initialized: boolean = false;

  private readonly defaultMean: number[] = [123.675, 116.28, 103.53];
  private readonly defaultStd: number[] = [58.395, 57.12, 57.375];

  constructor(
    onnxModel: string,
    modelInputSize: [number, number] = [384, 288],  // [height, width]
    toOpenpose: boolean = false,
    backend: 'onnxruntime' = 'onnxruntime',
    device: string = 'cpu'
  ) {
    super(onnxModel, modelInputSize, null, null, backend, device);
    this.toOpenpose = toOpenpose;
  }

  async init(): Promise<void> {
    // Web version - model path is direct URL
    await super.init();
    this.initialized = true;
  }

  async call(
    image: Uint8Array,
    imgWidth: number,
    imgHeight: number,
    bboxes: BBox[] = []
  ): Promise<{ keypoints: number[][]; scores: number[] }> {
    if (!this.initialized) {
      await this.init();
    }

    if (bboxes.length === 0) {
      bboxes = [{ x1: 0, y1: 0, x2: imgWidth, y2: imgHeight }];
    }

    const allKeypoints: number[][][] = [];
    const allScores: number[][] = [];

    for (const bbox of bboxes) {
      const { tensor, center, scale, inputSize } = this.preprocess(
        image,
        imgWidth,
        imgHeight,
        bbox
      );

      const outputs = await this.inference(tensor, inputSize);
      const { keypoints, scores } = this.postprocess(outputs[0].data as Float32Array, outputs[1].data as Float32Array, outputs[0].dims, outputs[1].dims, center, scale);

      allKeypoints.push(keypoints);
      allScores.push(scores);
    }

    // Flatten results
    const keypoints = allKeypoints.flat();
    const scores = allScores.flat();

    if (this.toOpenpose) {
      const converted = this.convertCocoToOpenpose(keypoints, scores);
      return converted;
    }

    return { keypoints, scores };
  }

  private preprocess(
    img: Uint8Array,
    imgWidth: number,
    imgHeight: number,
    bbox: BBox
  ): { tensor: Float32Array; center: [number, number]; scale: [number, number]; inputSize: [number, number] } {
    const [inputH, inputW] = this.modelInputSize;  // H=384, W=288
    
    // Center and scale from bbox with padding
    const center: [number, number] = [
      bbox.x1 + (bbox.x2 - bbox.x1) / 2,
      bbox.y1 + (bbox.y2 - bbox.y1) / 2,
    ];
    
    const bboxWidth = bbox.x2 - bbox.x1;
    const bboxHeight = bbox.y2 - bbox.y1;
    const bboxAspectRatio = bboxWidth / bboxHeight;
    const modelAspectRatio = inputW / inputH;
    
    let scaleW: number, scaleH: number;
    if (bboxAspectRatio > modelAspectRatio) {
      scaleW = bboxWidth * 1.25;
      scaleH = scaleW / modelAspectRatio;
    } else {
      scaleH = bboxHeight * 1.25;
      scaleW = scaleH * modelAspectRatio;
    }
    
    const scale: [number, number] = [scaleW, scaleH];

    // Create canvas for cropping
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    canvas.width = inputW;
    canvas.height = inputH;

    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, inputW, inputH);

    // Create source canvas from image data
    const srcCanvas = document.createElement('canvas');
    const srcCtx = srcCanvas.getContext('2d')!;
    srcCanvas.width = imgWidth;
    srcCanvas.height = imgHeight;
    
    const srcImageData = srcCtx.createImageData(imgWidth, imgHeight);
    srcImageData.data.set(img);
    srcCtx.putImageData(srcImageData, 0, 0);

    // Calculate source region
    const srcX = center[0] - scaleW / 2;
    const srcY = center[1] - scaleH / 2;

    // Draw cropped and scaled region
    ctx.drawImage(
      srcCanvas,
      srcX,
      srcY,
      scaleW,
      scaleH,
      0,
      0,
      inputW,
      inputH
    );

    const imageData = ctx.getImageData(0, 0, inputW, inputH);

    // Normalize with mean/std
    const data = new Float32Array(inputW * inputH * 3);
    for (let i = 0; i < imageData.data.length; i += 4) {
      const pixelIndex = i / 4;
      for (let c = 0; c < 3; c++) {
        const value = imageData.data[i + c];
        data[c * inputW * inputH + pixelIndex] = (value - this.defaultMean[c]) / this.defaultStd[c];
      }
    }

    return {
      tensor: data,
      center,
      scale,
      inputSize: [inputH, inputW],
    };
  }

  private postprocess(
    simccX: Float32Array,
    simccY: Float32Array,
    outputShapeX: number[],
    outputShapeY: number[],
    center: [number, number],
    scale: [number, number]
  ): { keypoints: number[][]; scores: number[] } {
    const numKeypoints = outputShapeX[1];
    const wx = outputShapeX[2];
    const wy = outputShapeY[2];

    const keypoints: number[][] = [];
    const scores: number[] = [];

    for (let k = 0; k < numKeypoints; k++) {
      // Find argmax for x
      let maxX = -Infinity;
      let argmaxX = 0;
      for (let i = 0; i < wx; i++) {
        const val = simccX[k * wx + i];
        if (val > maxX) {
          maxX = val;
          argmaxX = i;
        }
      }

      // Find argmax for y
      let maxY = -Infinity;
      let argmaxY = 0;
      for (let i = 0; i < wy; i++) {
        const val = simccY[k * wy + i];
        if (val > maxY) {
          maxY = val;
          argmaxY = i;
        }
      }

      const score = 0.5 * (maxX + maxY);

      // Normalize to [0, 1] and transform to original image coordinates
      const normX = argmaxX / wx;
      const normY = argmaxY / wy;

      const kptX = (normX - 0.5) * scale[0] + center[0];
      const kptY = (normY - 0.5) * scale[1] + center[1];

      keypoints.push([kptX, kptY]);
      scores.push(score);
    }

    return { keypoints, scores };
  }

  private convertCocoToOpenpose(
    keypoints: number[][],
    scores: number[]
  ): { keypoints: number[][]; scores: number[] } {
    // COCO 17 keypoints to OpenPose 18 keypoints mapping
    const cocoToOpenpose: number[] = [
      0,  // nose
      1,  // neck (average of shoulders)
      2,  // right_shoulder
      3,  // right_elbow
      4,  // right_wrist
      5,  // left_shoulder
      6,  // left_elbow
      7,  // left_wrist
      8,  // right_hip
      9,  // right_knee
      10, // right_ankle
      11, // left_hip
      12, // left_knee
      13, // left_ankle
      14, // right_eye
      15, // left_eye
      16, // left_ear
    ];

    const openposeKeypoints: number[][] = [];
    const openposeScores: number[] = [];

    for (let i = 0; i < 17; i++) {
      if (i === 1) {
        // Neck is average of shoulders
        const rightShoulder = keypoints[2];
        const leftShoulder = keypoints[5];
        openposeKeypoints.push([
          (rightShoulder[0] + leftShoulder[0]) / 2,
          (rightShoulder[1] + leftShoulder[1]) / 2,
        ]);
        openposeScores.push((scores[2] + scores[5]) / 2);
      } else {
        const cocoIdx = cocoToOpenpose[i];
        openposeKeypoints.push([...keypoints[cocoIdx]]);
        openposeScores.push(scores[cocoIdx]);
      }
    }

    return { keypoints: openposeKeypoints, scores: openposeScores };
  }
}
