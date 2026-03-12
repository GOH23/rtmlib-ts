/**
 * RTMPose3D model for 3D pose estimation
 * Extends RTMPose with Z-axis prediction
 * Based on rtmlib RTMPose3d class
 */

import { BaseTool } from '../core/base';
import { BBox, BackendType } from '../types/index';

export class RTMPose3D extends BaseTool {
  private toOpenpose: boolean;
  private simccSplitRatio: number = 2.0;
  private zRange: number = 2.1744869;
  private initialized: boolean = false;

  private readonly defaultMean: number[] = [123.675, 116.28, 103.53];
  private readonly defaultStd: number[] = [58.395, 57.12, 57.375];

  constructor(
    onnxModel: string,
    modelInputSize: [number, number] = [288, 384], // [width=288, height=384] - creates tensor [1,3,384,288]
    toOpenpose: boolean = false,
    backend: BackendType = 'webgpu',
    zRange?: number
  ) {
    super(onnxModel, modelInputSize, null, null, backend);
    this.toOpenpose = toOpenpose;
    if (zRange !== undefined) {
      this.zRange = zRange;
    }
  }

  async init(): Promise<void> {
    await super.init();
    this.initialized = true;
  }

  async call(
    image: Uint8Array,
    imgWidth: number,
    imgHeight: number,
    bboxes: BBox[] = []
  ): Promise<{
    keypoints: number[][][];
    scores: number[][];
    keypointsSimcc: number[][][];
    keypoints2d: number[][][];
  }> {
    if (!this.initialized) {
      await this.init();
    }

    if (bboxes.length === 0) {
      bboxes = [{ x1: 0, y1: 0, x2: imgWidth, y2: imgHeight }];
    }

    const allKeypoints: number[][][] = [];
    const allScores: number[][] = [];
    const allKeypointsSimcc: number[][][] = [];
    const allKeypoints2d: number[][][] = [];

    for (const bbox of bboxes) {
      const { tensor, center, scale, inputSize } = this.preprocess(
        image,
        imgWidth,
        imgHeight,
        bbox
      );

      const outputs = await this.inference(tensor, inputSize);
      const { keypoints, scores, keypointsSimcc, keypoints2d } = this.postprocess(
        outputs[0].data as Float32Array,
        outputs[1].data as Float32Array,
        outputs[2].data as Float32Array,
        outputs[0].dims,
        outputs[1].dims,
        outputs[2].dims,
        center,
        scale
      );

      allKeypoints.push(keypoints);
      allScores.push(scores);
      allKeypointsSimcc.push(keypointsSimcc);
      allKeypoints2d.push(keypoints2d);
    }

    return {
      keypoints: allKeypoints,
      scores: allScores,
      keypointsSimcc: allKeypointsSimcc,
      keypoints2d: allKeypoints2d,
    };
  }

  private preprocess(
    img: Uint8Array,
    imgWidth: number,
    imgHeight: number,
    bbox: BBox
  ): {
    tensor: Float32Array;
    center: [number, number];
    scale: [number, number];
    inputSize: [number, number];
  } {
    const [inputH, inputW] = this.modelInputSize;

    // Center and scale from bbox with padding (1.25 as in Python)
    const center: [number, number] = [
      bbox.x1 + (bbox.x2 - bbox.x1) / 2,
      bbox.y1 + (bbox.y2 - bbox.y1) / 2,
    ];

    const bboxWidth = bbox.x2 - bbox.x1;
    const bboxHeight = bbox.y2 - bbox.y1;
    const padding = 1.25;

    // Adjust scale to maintain aspect ratio
    const aspectRatio = inputW / inputH;
    const bboxAspectRatio = bboxWidth / bboxHeight;

    let scaleW: number, scaleH: number;
    if (bboxAspectRatio > aspectRatio) {
      scaleW = bboxWidth * padding;
      scaleH = scaleW / aspectRatio;
    } else {
      scaleH = bboxHeight * padding;
      scaleW = scaleH * aspectRatio;
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

    // Draw cropped and scaled region using warpAffine-like transformation
    this.warpAffine(ctx, srcCanvas, center, scale, inputW, inputH, srcX, srcY);

    const imageData = ctx.getImageData(0, 0, inputW, inputH);

    // Normalize with mean/std
    const data = new Float32Array(inputW * inputH * 3);
    for (let i = 0; i < imageData.data.length; i += 4) {
      const pixelIndex = i / 4;
      for (let c = 0; c < 3; c++) {
        const value = imageData.data[i + c];
        data[c * inputW * inputH + pixelIndex] =
          (value - this.defaultMean[c]) / this.defaultStd[c];
      }
    }

    return {
      tensor: data,
      center,
      scale,
      inputSize: [inputH, inputW],
    };
  }

  private warpAffine(
    ctx: CanvasRenderingContext2D,
    srcCanvas: HTMLCanvasElement,
    center: [number, number],
    scale: [number, number],
    dstWidth: number,
    dstHeight: number,
    srcX: number,
    srcY: number
  ): void {
    // Simple affine transform using canvas drawImage
    // For more accurate transformation, OpenCV bindings would be needed
    ctx.drawImage(srcCanvas, srcX, srcY, scale[0], scale[1], 0, 0, dstWidth, dstHeight);
  }

  private postprocess(
    simccX: Float32Array,
    simccY: Float32Array,
    simccZ: Float32Array,
    outputShapeX: number[],
    outputShapeY: number[],
    outputShapeZ: number[],
    center: [number, number],
    scale: [number, number]
  ): {
    keypoints: number[][];
    scores: number[];
    keypointsSimcc: number[][];
    keypoints2d: number[][];
  } {
    const numKeypoints = outputShapeX[1];
    const wx = outputShapeX[2];
    const wy = outputShapeY[2];
    const wz = outputShapeZ[2];

    const keypoints: number[][] = [];
    const scores: number[] = [];
    const keypointsSimcc: number[][] = [];
    const keypoints2d: number[][] = [];

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

      // Find argmax for z
      let maxZ = -Infinity;
      let argmaxZ = 0;
      for (let i = 0; i < wz; i++) {
        const val = simccZ[k * wz + i];
        if (val > maxZ) {
          maxZ = val;
          argmaxZ = i;
        }
      }

      // Score is max of x and y (as in Python)
      const score = maxX > maxY ? maxX : maxY;

      // Normalize to [0, 1] and transform to original image coordinates
      const normX = argmaxX / wx;
      const normY = argmaxY / wy;
      const normZ = argmaxZ / wz;

      // Apply split ratio
      const kptX = (normX - 0.5) * this.simccSplitRatio;
      const kptY = (normY - 0.5) * this.simccSplitRatio;
      const kptZ = (normZ - 0.5) * this.simccSplitRatio;

      // Convert Z to metric scale
      // Python uses model_input_size[-1] which is width (384) in (H, W) format
      // TypeScript uses modelInputSize[0] which is width (288) in [W, H] format
      const kptZMetric = (normZ / (this.modelInputSize[0] / 2) - 1) * this.zRange;

      // 3D keypoint
      keypoints.push([kptX, kptY, kptZMetric]);

      // SimCC coordinates (normalized)
      keypointsSimcc.push([normX, normY, normZ]);

      // 2D keypoint in original image coordinates
      const kpt2dX = normX * scale[0] + center[0] - 0.5 * scale[0];
      const kpt2dY = normY * scale[1] + center[1] - 0.5 * scale[1];
      keypoints2d.push([kpt2dX, kpt2dY]);

      scores.push(score);
    }

    return { keypoints, scores, keypointsSimcc, keypoints2d };
  }
}
