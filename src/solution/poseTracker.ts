/**
 * PoseTracker - tracks poses across frames with cached detections
 * Reduces detection frequency for better performance
 */

import { Wholebody } from './wholebody';
import { BBox } from '../types/index';

interface TrackedBox {
  bbox: BBox;
  age: number;
  lastSeen: number;
  id: number;
}

export class PoseTracker {
  private wholebody: Wholebody;
  private detFrequency: number;
  private cachedBoxes: TrackedBox[] = [];
  private frameCount: number = 0;
  private nextId: number = 0;

  constructor(
    WholebodyClass: typeof Wholebody,
    detFrequency: number = 7,
    toOpenpose: boolean = false,
    mode: 'performance' | 'lightweight' | 'balanced' = 'balanced',
    backend: 'onnxruntime' = 'onnxruntime',
    device: string = 'cpu'
  ) {
    this.detFrequency = detFrequency;
    this.wholebody = new WholebodyClass(
      null,
      [640, 640],
      null,
      [288, 384],
      mode,
      toOpenpose,
      backend,
      device
    );
  }

  async init(): Promise<void> {
    await this.wholebody.init();
  }

  async call(
    image: Uint8Array,
    imgWidth: number,
    imgHeight: number
  ): Promise<{ keypoints: number[][]; scores: number[] }> {
    this.frameCount++;

    // Run detection periodically
    if (this.frameCount % this.detFrequency === 0 || this.cachedBoxes.length === 0) {
      const result = await this.wholebody.call(image, imgWidth, imgHeight);
      this.updateCachedBoxes(result.keypoints, result.scores, imgWidth, imgHeight);
    }

    // Use cached boxes for pose estimation
    const bboxes = this.cachedBoxes.map((tb) => tb.bbox);
    const result = await this.wholebody.call(image, imgWidth, imgHeight, bboxes);

    // Clean up old boxes
    this.cleanupCachedBoxes();

    return result;
  }

  private updateCachedBoxes(
    keypoints: number[][],
    scores: number[],
    imgWidth: number,
    imgHeight: number
  ): void {
    // Simple tracking: create new boxes from keypoints
    const newBoxes: TrackedBox[] = [];

    for (let i = 0; i < keypoints.length; i += 17) {
      const instanceKeypoints = keypoints.slice(i, Math.min(i + 17, keypoints.length));
      const instanceScores = scores.slice(i, Math.min(i + 17, scores.length));

      // Calculate bounding box from keypoints
      let minX = imgWidth;
      let minY = imgHeight;
      let maxX = 0;
      let maxY = 0;

      for (let j = 0; j < instanceKeypoints.length; j++) {
        if (instanceScores[j] > 0.3) {
          const [x, y] = instanceKeypoints[j];
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }

      if (maxX > minX && maxY > minY) {
        // Add padding
        const padding = 0.25;
        const width = maxX - minX;
        const height = maxY - minY;
        const paddedWidth = width * (1 + padding);
        const paddedHeight = height * (1 + padding);

        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;

        const x1 = Math.max(0, centerX - paddedWidth / 2);
        const y1 = Math.max(0, centerY - paddedHeight / 2);
        const x2 = Math.min(imgWidth, centerX + paddedWidth / 2);
        const y2 = Math.min(imgHeight, centerY + paddedHeight / 2);

        // Try to match with existing boxes
        let matched = false;
        for (const cachedBox of this.cachedBoxes) {
          const iou = this.calculateIoU(
            [x1, y1, x2, y2],
            [cachedBox.bbox.x1, cachedBox.bbox.y1, cachedBox.bbox.x2, cachedBox.bbox.y2]
          );

          if (iou > 0.3) {
            cachedBox.bbox = { x1, y1, x2, y2 };
            cachedBox.age = 0;
            cachedBox.lastSeen = this.frameCount;
            newBoxes.push(cachedBox);
            matched = true;
            break;
          }
        }

        if (!matched) {
          newBoxes.push({
            bbox: { x1, y1, x2, y2 },
            age: 0,
            lastSeen: this.frameCount,
            id: this.nextId++,
          });
        }
      }
    }

    this.cachedBoxes = newBoxes;
  }

  private calculateIoU(box1: number[], box2: number[]): number {
    const x1 = Math.max(box1[0], box2[0]);
    const y1 = Math.max(box1[1], box2[1]);
    const x2 = Math.min(box1[2], box2[2]);
    const y2 = Math.min(box1[3], box2[3]);

    const interWidth = Math.max(0, x2 - x1);
    const interHeight = Math.max(0, y2 - y1);
    const interArea = interWidth * interHeight;

    const box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    const box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);

    const unionArea = box1Area + box2Area - interArea;
    return unionArea > 0 ? interArea / unionArea : 0;
  }

  private cleanupCachedBoxes(): void {
    // Remove boxes that haven't been seen for too long
    const maxAge = this.detFrequency * 3;
    this.cachedBoxes = this.cachedBoxes.filter(
      (box) => this.frameCount - box.lastSeen < maxAge
    );
  }
}
