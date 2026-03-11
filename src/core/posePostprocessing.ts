/**
 * Post-processing utilities for pose estimation
 */

export function getSimccMaximum(
  simccX: Float32Array,
  simccY: Float32Array
): { locations: number[]; scores: number[] } {
  const numKeypoints = simccX.length / 2; // Assuming split_ratio = 2
  
  const locations: number[] = [];
  const scores: number[] = [];
  
  for (let i = 0; i < numKeypoints; i++) {
    // Find argmax for x
    let maxX = -Infinity;
    let argmaxX = 0;
    for (let j = 0; j < 2; j++) {
      const val = simccX[i * 2 + j];
      if (val > maxX) {
        maxX = val;
        argmaxX = j;
      }
    }
    
    // Find argmax for y
    let maxY = -Infinity;
    let argmaxY = 0;
    for (let j = 0; j < 2; j++) {
      const val = simccY[i * 2 + j];
      if (val > maxY) {
        maxY = val;
        argmaxY = j;
      }
    }
    
    locations.push(argmaxX, argmaxY);
    scores.push((maxX + maxY) / 2);
  }
  
  return { locations, scores };
}

export function convertCocoToOpenpose(
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
    16, // right_ear
    17, // left_ear
  ];
  
  const openposeKeypoints: number[][] = [];
  const openposeScores: number[] = [];
  
  for (let i = 0; i < 18; i++) {
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
