/**
 * Post-processing utilities for object detection
 */

import { BBox } from '../types/index.js';

export function multiclassNms(
  boxes: number[][],
  scores: number[][],
  nmsThr: number,
  scoreThr: number
): { boxes: number[][]; scores: number[]; classIds: number[] } | null {
  const numClasses = scores[0].length;
  const numBoxes = boxes.length;
  
  const allDetections: Array<{ box: number[]; score: number; classId: number }> = [];
  
  for (let c = 0; c < numClasses; c++) {
    for (let i = 0; i < numBoxes; i++) {
      if (scores[i][c] > scoreThr) {
        allDetections.push({
          box: boxes[i],
          score: scores[i][c],
          classId: c,
        });
      }
    }
  }
  
  allDetections.sort((a, b) => b.score - a.score);
  
  const keep: typeof allDetections = [];
  
  while (allDetections.length > 0) {
    const current = allDetections.shift()!;
    keep.push(current);
    
    for (let i = allDetections.length - 1; i >= 0; i--) {
      const other = allDetections[i];
      if (current.classId !== other.classId) continue;
      
      const iou = calculateIoU(current.box, other.box);
      if (iou > nmsThr) {
        allDetections.splice(i, 1);
      }
    }
  }
  
  if (keep.length === 0) return null;
  
  return {
    boxes: keep.map(d => d.box),
    scores: keep.map(d => d.score),
    classIds: keep.map(d => d.classId),
  };
}

function calculateIoU(box1: number[], box2: number[]): number {
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

export function nms(boxes: number[][], scores: number[], nmsThr: number): number[] {
  const indices: number[] = [];
  const sortedIndices = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
  
  while (sortedIndices.length > 0) {
    const current = sortedIndices.shift()!;
    indices.push(current);
    
    for (let i = sortedIndices.length - 1; i >= 0; i--) {
      const other = sortedIndices[i];
      const iou = calculateIoU(boxes[current], boxes[other]);
      if (iou > nmsThr) {
        sortedIndices.splice(i, 1);
      }
    }
  }
  
  return indices;
}
