/**
 * YOLO12 object detection model
 * Based on YOLO12 architecture for person detection
 * Compatible with Ultralytics YOLOv12 ONNX export
 * Uses onnxruntime-web for inference
 */

import { BaseTool } from '../core/base';
import { BBox, Detection } from '../types/index';

export class YOLO12 extends BaseTool {
  private nmsThr: number;
  public scoreThr: number;
  private initialized = false;
  private paddingX = 0;
  private paddingY = 0;
  private scaleX = 1;
  private scaleY = 1;

  constructor(
    modelPath: string,
    modelInputSize: [number, number] = [640, 640],
    nmsThr: number = 0.45,
    scoreThr: number = 0.5
  ) {
    super(modelPath, modelInputSize);
    this.nmsThr = nmsThr;
    this.scoreThr = scoreThr;
  }

  async init(): Promise<void> {
    await super.init();
    this.initialized = true;
  }

  async call(
    image: Uint8Array,
    imgWidth: number,
    imgHeight: number
  ): Promise<Detection[]> {
    if (!this.initialized) {
      await this.init();
    }

    const { paddedImg } = this.preprocess(image, imgWidth, imgHeight);
    const outputs = await this.inference(paddedImg);

    // YOLO12 output format: [1, num_boxes, 6] where 6 = [x1, y1, x2, y2, score, class_id]
    const detOutput = outputs[0];
    const detShape = detOutput.dims;

    if (detShape.length !== 3 || detShape[2] !== 6 || detOutput.type !== 'float32') {
      console.error(`YOLO12: Unexpected output shape [${detShape}] or type ${detOutput.type}`);
      return [];
    }

    const detArray = detOutput.data as Float32Array;
    const numBoxes = detShape[1];
    const detections: Detection[] = [];

    for (let i = 0; i < numBoxes; i++) {
      const baseIdx = i * 6;
      let x1 = detArray[baseIdx];
      let y1 = detArray[baseIdx + 1];
      let x2 = detArray[baseIdx + 2];
      let y2 = detArray[baseIdx + 3];
      const score = detArray[baseIdx + 4];
      const classId = detArray[baseIdx + 5];

      // Filter by score threshold and class (0 = person in COCO)
      if (score < this.scoreThr || classId !== 0) {
        continue;
      }

      // Transform from padded coordinates to original image coordinates
      const transformedX1 = (x1 - this.paddingX) * this.scaleX;
      const transformedY1 = (y1 - this.paddingY) * this.scaleY;
      const transformedX2 = (x2 - this.paddingX) * this.scaleX;
      const transformedY2 = (y2 - this.paddingY) * this.scaleY;

      // Validate box coordinates
      if (transformedX1 >= transformedX2 || transformedY1 >= transformedY2) {
        continue;
      }

      detections.push({
        bbox: {
          x1: Math.max(0, transformedX1),
          y1: Math.max(0, transformedY1),
          x2: Math.min(imgWidth, transformedX2),
          y2: Math.min(imgHeight, transformedY2),
        },
        score,
        classId: Math.round(classId),
      });
    }

    // Apply NMS
    return this.applyNms(detections, this.nmsThr);
  }

  private preprocess(
    img: Uint8Array,
    imgWidth: number,
    imgHeight: number
  ): { paddedImg: Float32Array; ratio: number } {
    const [inputH, inputW] = this.modelInputSize;

    // Create canvas for padded image (black background)
    const paddedImg = new Uint8Array(inputH * inputW * 3).fill(0);

    // Calculate scaling and positioning to maintain aspect ratio
    const aspectRatio = imgWidth / imgHeight;
    const targetAspectRatio = inputW / inputH;

    let drawWidth: number, drawHeight: number;

    if (aspectRatio > targetAspectRatio) {
      // Image is wider - fit to width, add padding top/bottom
      drawWidth = inputW;
      drawHeight = Math.floor(inputW / aspectRatio);
      this.paddingX = 0;
      this.paddingY = (inputH - drawHeight) / 2;
    } else {
      // Image is taller - fit to height, add padding left/right
      drawHeight = inputH;
      drawWidth = Math.floor(inputH * aspectRatio);
      this.paddingX = (inputW - drawWidth) / 2;
      this.paddingY = 0;
    }

    // Calculate scale factors
    this.scaleX = imgWidth / drawWidth;
    this.scaleY = imgHeight / drawHeight;

    // Resize image onto padded canvas (nearest neighbor)
    for (let y = 0; y < drawHeight; y++) {
      for (let x = 0; x < drawWidth; x++) {
        const srcX = Math.floor(x * this.scaleX);
        const srcY = Math.floor(y * this.scaleY);
        const dstX = Math.floor(x + this.paddingX);
        const dstY = Math.floor(y + this.paddingY);

        for (let c = 0; c < 3; c++) {
          paddedImg[(dstY * inputW + dstX) * 3 + c] = img[(srcY * imgWidth + srcX) * 3 + c];
        }
      }
    }

    // Normalize to [0, 1] and convert to float32
    const floatImg = new Float32Array(paddedImg.length);
    for (let i = 0; i < paddedImg.length; i++) {
      floatImg[i] = paddedImg[i] / 255.0;
    }

    // Transpose HWC to CHW
    const transposed = new Float32Array(3 * inputH * inputW);
    for (let c = 0; c < 3; c++) {
      for (let h = 0; h < inputH; h++) {
        for (let w = 0; w < inputW; w++) {
          transposed[c * inputH * inputW + h * inputW + w] =
            floatImg[h * inputW * 3 + w * 3 + c];
        }
      }
    }

    return { paddedImg: transposed, ratio: 1 };
  }

  private applyNms(detections: Detection[], iouThreshold: number): Detection[] {
    if (detections.length === 0) {
      return [];
    }

    // Sort by score descending
    detections.sort((a, b) => b.score - a.score);

    const selected: Detection[] = [];
    const used: boolean[] = new Array(detections.length).fill(false);

    for (let i = 0; i < detections.length; i++) {
      if (used[i]) {
        continue;
      }

      selected.push(detections[i]);
      used[i] = true;

      const boxA = detections[i].bbox;

      for (let j = i + 1; j < detections.length; j++) {
        if (used[j]) {
          continue;
        }

        const boxB = detections[j].bbox;

        // Calculate IoU
        const x1 = Math.max(boxA.x1, boxB.x1);
        const y1 = Math.max(boxA.y1, boxB.y1);
        const x2 = Math.min(boxA.x2, boxB.x2);
        const y2 = Math.min(boxA.y2, boxB.y2);

        const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const areaA = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1);
        const areaB = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1);
        const union = areaA + areaB - intersection;

        const iou = union > 0 ? intersection / union : 0;

        if (iou <= iouThreshold) {
          used[j] = true;
        }
      }
    }

    return selected;
  }
}
