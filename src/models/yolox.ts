/**
 * YOLOX object detection model
 * Based on https://github.com/IDEA-Research/DWPose/blob/opencv_onnx/ControlNet-v1-1-nightly/annotator/dwpose/cv_ox_det.py
 */

import { BaseTool } from '../core/base';
import { multiclassNms } from '../core/postprocessing';
import { BBox, BackendType } from '../types/index';

export class YOLOX extends BaseTool {
  private nmsThr: number;
  public scoreThr: number;
  private initialized: boolean = false;

  constructor(
    onnxModel: string,
    modelInputSize: [number, number] = [640, 640],
    nmsThr: number = 0.45,
    scoreThr: number = 0.3,  // Lower default threshold
    backend: BackendType = 'webgpu'
  ) {
    super(onnxModel, modelInputSize, null, null, backend);
    this.nmsThr = nmsThr;
    this.scoreThr = scoreThr;
  }

  async init(): Promise<void> {
    // Web version - model path is direct URL
    await super.init();
    this.initialized = true;
  }

  async call(image: Uint8Array, imgWidth: number, imgHeight: number): Promise<BBox[]> {
    if (!this.initialized) {
      await this.init();
    }

    const { paddedImg, ratio } = this.preprocess(image, imgWidth, imgHeight);
    const outputs = await this.inference(paddedImg);

    console.log(`YOLOX: Got ${outputs.length} outputs`);
    for (let i = 0; i < outputs.length; i++) {
      console.log(`  Output[${i}]: dims=[${outputs[i].dims}], type=${outputs[i].type}`);
    }

    // For end2end YOLOX with built-in NMS:
    // Output 0: [1, num_dets, 5] where 5 = [x1, y1, x2, y2, score]
    // Output 1: [1, num_dets] or [1, 1] with count
    
    const detOutput = outputs[0];
    const detShape = detOutput.dims; // [1, num_dets, 5]
    
    console.log(`YOLOX: detShape=[${detShape}], ratio=${ratio}`);
    
    if (detShape.length === 3 && detShape[2] === 5 && detOutput.type === 'float32') {
      const detArray = detOutput.data as Float32Array;
      const numDets = detShape[1];
      const boxes: BBox[] = [];
      
      console.log(`YOLOX: Raw detections (first 5):`);
      for (let i = 0; i < Math.min(5, numDets); i++) {
        const baseIdx = i * 5;
        const x1 = detArray[baseIdx];
        const y1 = detArray[baseIdx + 1];
        const x2 = detArray[baseIdx + 2];
        const y2 = detArray[baseIdx + 3];
        const score = detArray[baseIdx + 4];
        console.log(`  [${i}] raw=[${x1.toFixed(2)}, ${y1.toFixed(2)}, ${x2.toFixed(2)}, ${y2.toFixed(2)}] score=${score.toFixed(4)}`);
      }
      
      for (let i = 0; i < numDets; i++) {
        const baseIdx = i * 5;
        let x1 = detArray[baseIdx];
        let y1 = detArray[baseIdx + 1];
        let x2 = detArray[baseIdx + 2];
        let y2 = detArray[baseIdx + 3];
        const score = detArray[baseIdx + 4];
        
        // Scale to original image
        x1 /= ratio;
        y1 /= ratio;
        x2 /= ratio;
        y2 /= ratio;
        
        // Python uses score > 0.3 threshold
        if (score > 0.3 && x2 > x1 && y2 > y1) {
          boxes.push({ x1, y1, x2, y2 });
          console.log(`  [${i}] ACCEPTED: [${x1.toFixed(1)}, ${y1.toFixed(1)}, ${x2.toFixed(1)}, ${y2.toFixed(1)}] score=${score.toFixed(3)}`);
        } else if (score > 0.1) {
          console.log(`  [${i}] rejected (score=${score.toFixed(3)}): [${x1.toFixed(1)}, ${y1.toFixed(1)}, ${x2.toFixed(1)}, ${y2.toFixed(1)}]`);
        }
      }
      
      console.log(`YOLOX: Found ${boxes.length} boxes`);
      return boxes;
    }

    return [];
  }

  private preprocess(
    img: Uint8Array,
    imgWidth: number,
    imgHeight: number
  ): { paddedImg: Float32Array; ratio: number } {
    const [inputH, inputW] = this.modelInputSize;

    let paddedImg: Uint8Array;
    let ratio: number;

    if (imgHeight === inputH && imgWidth === inputW) {
      paddedImg = img;
      ratio = 1.0;
    } else {
      paddedImg = new Uint8Array(inputH * inputW * 3).fill(114);

      ratio = Math.min(inputH / imgHeight, inputW / imgWidth);
      const resizedW = Math.floor(imgWidth * ratio);
      const resizedH = Math.floor(imgHeight * ratio);

      // Resize image (simple nearest neighbor for now)
      for (let y = 0; y < resizedH; y++) {
        for (let x = 0; x < resizedW; x++) {
          const srcX = Math.floor(x / ratio);
          const srcY = Math.floor(y / ratio);
          for (let c = 0; c < 3; c++) {
            paddedImg[(y * inputW + x) * 3 + c] = img[(srcY * imgWidth + srcX) * 3 + c];
          }
        }
      }
    }

    // YOLOX uses simple normalization to [0, 1]
    // Convert to float32 and normalize to [0, 1]
    // Try BGR format (OpenCV standard)
    const floatImg = new Float32Array(paddedImg.length);
    for (let i = 0; i < paddedImg.length; i += 3) {
      // Swap RGB to BGR
      floatImg[i] = paddedImg[i + 2] / 255.0;     // B
      floatImg[i + 1] = paddedImg[i + 1] / 255.0; // G
      floatImg[i + 2] = paddedImg[i] / 255.0;     // R
    }

    // Transpose HWC to CHW
    const transposed = new Float32Array(inputH * inputW * 3);
    for (let c = 0; c < 3; c++) {
      for (let h = 0; h < inputH; h++) {
        for (let w = 0; w < inputW; w++) {
          transposed[c * inputH * inputW + h * inputW + w] = floatImg[h * inputW * 3 + w * 3 + c];
        }
      }
    }

    console.log(`YOLOX preprocess: input ${imgWidth}x${imgHeight} -> ${inputW}x${inputH}, ratio=${ratio} (BGR)`);

    return { paddedImg: transposed, ratio };
  }

  private postprocess(outputs: any, ratio: number): BBox[] {
    const outputArray = new Float32Array(outputs.data);
    const outputShape = outputs.dims;
    
    console.log(`YOLOX output shape: [${outputShape}], ratio: ${ratio}`);
    console.log(`First 20 values: ${Array.from(outputArray.slice(0, 20)).map(v => v.toFixed(4)).join(', ')}`);
    
    // outputShape: [1, num_boxes, 5] or [1, num_boxes, 6]
    // For YOLOX with NMS: [batch, num_dets, 5] where 5 = [x1, y1, x2, y2, score]
    
    if (outputShape.length === 3 && outputShape[2] >= 5) {
      const numBoxes = outputShape[1];
      const boxes: BBox[] = [];
      const hasClassInfo = outputShape[2] >= 6;
      
      console.log(`Processing ${numBoxes} boxes, hasClassInfo: ${hasClassInfo}`);
      
      for (let i = 0; i < numBoxes; i++) {
        const baseIdx = i * outputShape[2];
        const score = outputArray[baseIdx + 4];
        
        // Filter by score threshold
        if (score < this.scoreThr) continue;
        
        // Check class if available
        if (hasClassInfo) {
          const classId = outputArray[baseIdx + 5];
          if (classId !== 0) continue; // Only person class
        }
        
        const x1 = outputArray[baseIdx] / ratio;
        const y1 = outputArray[baseIdx + 1] / ratio;
        const x2 = outputArray[baseIdx + 2] / ratio;
        const y2 = outputArray[baseIdx + 3] / ratio;
        
        // Validate box coordinates
        if (x1 >= x2 || y1 >= y2) continue;
        if (x2 < 0 || y2 < 0 || x1 > this.modelInputSize[1] / ratio || y1 > this.modelInputSize[0] / ratio) continue;
        
        console.log(`Found box: [${x1.toFixed(1)}, ${y1.toFixed(1)}, ${x2.toFixed(1)}, ${y2.toFixed(1)}] score: ${score.toFixed(3)}`);
        
        boxes.push({
          x1: Math.max(0, x1),
          y1: Math.max(0, y1),
          x2: Math.min(outputShape[1] * ratio, x2),
          y2: Math.min(outputShape[0] * ratio, y2),
        });
      }
      
      console.log(`Total boxes found: ${boxes.length}`);
      return boxes;
    }
    
    return [];
  }
}
