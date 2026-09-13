/**
 * Drawing utilities for visualization
 * Based on rtmlib Python library
 */

import { SKELETON_EDGES } from '../core/instanthmrGeometry';
import type { InstantHMRPerson } from '../solution/pose3dDetector';

interface KeypointInfo {
  name: string;
  id: number;
  color: number[];
}

interface SkeletonInfo {
  link: [string, string];
  color: number[];
}

interface SkeletonDict {
  keypoint_info: Record<number, KeypointInfo>;
  skeleton_info: Record<number, SkeletonInfo>;
}

/**
 * Draw bounding boxes on image
 */
export function drawBbox(
  img: Uint8Array,
  width: number,
  height: number,
  bboxes: Array<[number, number, number, number]>,
  color: [number, number, number] = [0, 255, 0]
): Uint8Array {
  const result = new Uint8Array(img);
  
  for (const bbox of bboxes) {
    const [x1, y1, x2, y2] = bbox;
    
    // Draw top and bottom horizontal lines
    for (let x = Math.floor(x1); x < Math.floor(x2); x++) {
      for (let t = 0; t < 2; t++) {
        const yTop = Math.floor(y1) + t;
        const yBottom = Math.floor(y2) - t;
        if (yTop >= 0 && yTop < height && x >= 0 && x < width) {
          const idx = (yTop * width + x) * 3;
          result[idx] = color[2];
          result[idx + 1] = color[1];
          result[idx + 2] = color[0];
        }
        if (yBottom >= 0 && yBottom < height && x >= 0 && x < width) {
          const idx = (yBottom * width + x) * 3;
          result[idx] = color[2];
          result[idx + 1] = color[1];
          result[idx + 2] = color[0];
        }
      }
    }
    
    // Draw left and right vertical lines
    for (let y = Math.floor(y1); y < Math.floor(y2); y++) {
      for (let t = 0; t < 2; t++) {
        const xLeft = Math.floor(x1) + t;
        const xRight = Math.floor(x2) - t;
        if (y >= 0 && y < height && xLeft >= 0 && xLeft < width) {
          const idx = (y * width + xLeft) * 3;
          result[idx] = color[2];
          result[idx + 1] = color[1];
          result[idx + 2] = color[0];
        }
        if (y >= 0 && y < height && xRight >= 0 && xRight < width) {
          const idx = (y * width + xRight) * 3;
          result[idx] = color[2];
          result[idx + 1] = color[1];
          result[idx + 2] = color[0];
        }
      }
    }
  }
  
  return result;
}

/**
 * Draw skeleton on image
 */
export function drawSkeleton(
  img: Uint8Array,
  width: number,
  height: number,
  keypoints: number[][],
  scores: number[],
  openposeSkeleton: boolean = false,
  kptThr: number = 0.5,
  radius: number = 2,
  lineWidth: number = 2
): Uint8Array {
  const numKeypoints = keypoints.length;
  
  // Handle empty keypoints - return a copy of the image
  if (numKeypoints === 0) {
    console.log('No keypoints to draw');
    return new Uint8Array(img);
  }
  
  let skeletonName: string;
  
  if (openposeSkeleton) {
    if (numKeypoints === 18) {
      skeletonName = 'openpose18';
    } else if (numKeypoints === 134 || numKeypoints === 133) {
      skeletonName = 'openpose134';
    } else if (numKeypoints === 26) {
      skeletonName = 'halpe26';
    } else {
      throw new Error(`Unsupported openpose skeleton with ${numKeypoints} keypoints`);
    }
  } else {
    if (numKeypoints === 17) {
      skeletonName = 'coco17';
    } else if (numKeypoints === 133 || numKeypoints === 134) {
      skeletonName = 'coco133';
    } else if (numKeypoints === 21) {
      skeletonName = 'hand21';
    } else if (numKeypoints === 26) {
      skeletonName = 'halpe26';
    } else {
      throw new Error(`Unsupported mmpose skeleton with ${numKeypoints} keypoints`);
    }
  }
  
  const skeletonDict = getSkeletonDict(skeletonName);
  
  // Single instance - keypoints is 2D array [N, 2]
  img = drawMmpose(
    img,
    width,
    height,
    keypoints,
    scores,
    skeletonDict.keypoint_info,
    skeletonDict.skeleton_info,
    kptThr,
    radius,
    lineWidth
  );
  
  return img;
}

function getSkeletonDict(name: string): SkeletonDict {
  // Import skeleton configs dynamically
  switch (name) {
    case 'coco17':
      return {
        keypoint_info: {
          0: { name: 'nose', id: 0, color: [51, 255, 255] },
          1: { name: 'left_eye', id: 1, color: [51, 255, 255] },
          2: { name: 'right_eye', id: 2, color: [51, 255, 255] },
          3: { name: 'left_ear', id: 3, color: [51, 255, 255] },
          4: { name: 'right_ear', id: 4, color: [51, 255, 255] },
          5: { name: 'left_shoulder', id: 5, color: [255, 51, 255] },
          6: { name: 'right_shoulder', id: 6, color: [255, 51, 255] },
          7: { name: 'left_elbow', id: 7, color: [255, 51, 255] },
          8: { name: 'right_elbow', id: 8, color: [255, 51, 255] },
          9: { name: 'left_wrist', id: 9, color: [255, 51, 255] },
          10: { name: 'right_wrist', id: 10, color: [255, 51, 255] },
          11: { name: 'left_hip', id: 11, color: [255, 255, 51] },
          12: { name: 'right_hip', id: 12, color: [255, 255, 51] },
          13: { name: 'left_knee', id: 13, color: [255, 255, 51] },
          14: { name: 'right_knee', id: 14, color: [255, 255, 51] },
          15: { name: 'left_ankle', id: 15, color: [255, 255, 51] },
          16: { name: 'right_ankle', id: 16, color: [255, 255, 51] },
        },
        skeleton_info: {
          0: { link: ['left_ankle', 'left_knee'], color: [255, 51, 255] },
          1: { link: ['left_knee', 'left_hip'], color: [255, 51, 255] },
          2: { link: ['left_hip', 'right_hip'], color: [255, 255, 51] },
          3: { link: ['right_hip', 'right_knee'], color: [255, 51, 255] },
          4: { link: ['right_knee', 'right_ankle'], color: [255, 51, 255] },
          5: { link: ['left_hip', 'left_shoulder'], color: [255, 255, 51] },
          6: { link: ['left_shoulder', 'left_elbow'], color: [255, 255, 51] },
          7: { link: ['left_elbow', 'left_wrist'], color: [255, 255, 51] },
          8: { link: ['left_hip', 'right_shoulder'], color: [255, 255, 51] },
          9: { link: ['right_shoulder', 'right_elbow'], color: [255, 255, 51] },
          10: { link: ['right_elbow', 'right_wrist'], color: [255, 255, 51] },
          11: { link: ['left_shoulder', 'right_shoulder'], color: [255, 255, 51] },
          12: { link: ['nose', 'left_shoulder'], color: [255, 255, 51] },
          13: { link: ['nose', 'right_shoulder'], color: [255, 255, 51] },
          14: { link: ['nose', 'left_eye'], color: [255, 255, 51] },
          15: { link: ['left_eye', 'right_eye'], color: [255, 255, 51] },
          16: { link: ['right_eye', 'nose'], color: [255, 255, 51] },
          17: { link: ['left_eye', 'left_ear'], color: [255, 255, 51] },
          18: { link: ['right_eye', 'right_ear'], color: [255, 255, 51] },
          19: { link: ['left_ear', 'left_shoulder'], color: [255, 255, 51] },
          20: { link: ['right_ear', 'right_shoulder'], color: [255, 255, 51] },
        },
      };
    case 'coco133':
      // For 133 keypoints, use simplified body skeleton
      return {
        keypoint_info: Object.fromEntries(
          Array.from({ length: 133 }, (_, i) => [
            i,
            { name: `kp_${i}`, id: i, color: [255, 255, 255] }
          ])
        ),
        skeleton_info: {
          0: { link: ['kp_15', 'kp_13'], color: [255, 51, 255] },
          1: { link: ['kp_13', 'kp_11'], color: [255, 51, 255] },
          2: { link: ['kp_11', 'kp_12'], color: [255, 255, 51] },
          3: { link: ['kp_12', 'kp_14'], color: [255, 51, 255] },
          4: { link: ['kp_14', 'kp_16'], color: [255, 51, 255] },
          5: { link: ['kp_11', 'kp_5'], color: [255, 255, 51] },
          6: { link: ['kp_5', 'kp_7'], color: [255, 255, 51] },
          7: { link: ['kp_7', 'kp_9'], color: [255, 255, 51] },
          8: { link: ['kp_12', 'kp_6'], color: [255, 255, 51] },
          9: { link: ['kp_6', 'kp_8'], color: [255, 255, 51] },
          10: { link: ['kp_8', 'kp_10'], color: [255, 255, 51] },
          11: { link: ['kp_5', 'kp_6'], color: [255, 255, 51] },
          12: { link: ['kp_0', 'kp_5'], color: [255, 255, 51] },
          13: { link: ['kp_0', 'kp_6'], color: [255, 255, 51] },
          14: { link: ['kp_0', 'kp_1'], color: [255, 255, 51] },
          15: { link: ['kp_1', 'kp_2'], color: [255, 255, 51] },
          16: { link: ['kp_2', 'kp_0'], color: [255, 255, 51] },
        },
      };
    default:
      throw new Error(`Unknown skeleton type: ${name}`);
  }
}

/**
 * Draw MMPose-style skeleton
 */
function drawMmpose(
  img: Uint8Array,
  width: number,
  height: number,
  keypoints: number[][],
  scores: number[],
  keypointInfo: Record<number, KeypointInfo>,
  skeletonInfo: Record<number, SkeletonInfo>,
  kptThr: number = 0.5,
  radius: number = 2,
  lineWidth: number = 2
): Uint8Array {
  const result = new Uint8Array(img);
  const visKpt = scores.map((s) => s >= kptThr);
  
  // Build keypoint name to id mapping
  const linkDict: Record<string, number> = {};
  
  // Draw keypoints
  for (const [idStr, kptInfo] of Object.entries(keypointInfo)) {
    const id = parseInt(idStr);
    const kptColor = kptInfo.color;
    linkDict[kptInfo.name] = kptInfo.id;
    
    if (id >= keypoints.length) continue;
    
    const kpt = keypoints[id];
    
    if (visKpt[id]) {
      drawCircle(
        result,
        width,
        height,
        kpt[0],
        kpt[1],
        radius,
        kptColor
      );
    }
  }
  
  // Draw skeleton links
  for (const skeInfo of Object.values(skeletonInfo)) {
    const [link0, link1] = skeInfo.link;
    const pt0 = linkDict[link0];
    const pt1 = linkDict[link1];
    
    if (pt0 === undefined || pt1 === undefined) continue;
    if (!visKpt[pt0] || !visKpt[pt1]) continue;
    
    const linkColor = skeInfo.color;
    const kpt0 = keypoints[pt0];
    const kpt1 = keypoints[pt1];
    
    drawLine(
      result,
      width,
      height,
      kpt0[0],
      kpt0[1],
      kpt1[0],
      kpt1[1],
      linkColor,
      lineWidth
    );
  }
  
  return result;
}

/**
 * Draw a circle on the image
 */
function drawCircle(
  img: Uint8Array,
  width: number,
  height: number,
  cx: number,
  cy: number,
  radius: number,
  color: number[]
): void {
  const x0 = Math.max(0, Math.floor(cx - radius));
  const x1 = Math.min(width, Math.ceil(cx + radius));
  const y0 = Math.max(0, Math.floor(cy - radius));
  const y1 = Math.min(height, Math.ceil(cy + radius));
  
  const rSquared = radius * radius;
  
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const dx = x - cx;
      const dy = y - cy;
      if (dx * dx + dy * dy <= rSquared) {
        const idx = (y * width + x) * 3;
        img[idx] = color[2];
        img[idx + 1] = color[1];
        img[idx + 2] = color[0];
      }
    }
  }
}

/**
 * Draw a line on the image using Bresenham's algorithm
 */
function drawLine(
  img: Uint8Array,
  width: number,
  height: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: number[],
  thickness: number = 2
): void {
  let x0i = Math.round(x0);
  let y0i = Math.round(y0);
  const x1i = Math.round(x1);
  const y1i = Math.round(y1);

  const dx = Math.abs(x1i - x0i);
  const dy = Math.abs(y1i - y0i);
  const sx = x0i < x1i ? 1 : -1;
  const sy = y0i < y1i ? 1 : -1;
  let err = dx - dy;

  // Draw with thickness
  const halfThickness = Math.floor(thickness / 2);

  while (true) {
    for (let dy_t = -halfThickness; dy_t <= halfThickness; dy_t++) {
      for (let dx_t = -halfThickness; dx_t <= halfThickness; dx_t++) {
        const x = x0i + dx_t;
        const y = y0i + dy_t;
        if (x >= 0 && x < width && y >= 0 && y < height) {
          const idx = (y * width + x) * 3;
          img[idx] = color[2];
          img[idx + 1] = color[1];
          img[idx + 2] = color[0];
        }
      }
    }

    if (x0i === x1i && y0i === y1i) break;
    const e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x0i += sx;
    }
    if (e2 < dx) {
      err += dx;
      y0i += sy;
    }
  }
}

/**
 * Draw detections on HTML Canvas
 * @param ctx - Canvas 2D context
 * @param detections - Array of detected objects
 * @param color - Base color for boxes (default: green)
 */
export function drawDetectionsOnCanvas(
  ctx: CanvasRenderingContext2D,
  detections: Array<{
    bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number };
    className?: string;
  }>,
  color: string = '#00ff00'
): void {
  detections.forEach((det, idx) => {
    const { bbox } = det;
    const label = det.className ? `${det.className} ${(bbox.confidence * 100).toFixed(0)}%` : `${(bbox.confidence * 100).toFixed(0)}%`;
    const boxColor = Array.isArray(color) ? color : color;
    const hueColor = typeof color === 'string' && color.startsWith('hsl') ? color : `hsl(${idx * 60}, 80%, 50%)`;

    // Draw bounding box
    ctx.strokeStyle = hueColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

    // Draw label background
    ctx.font = 'bold 12px sans-serif';
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = hueColor;
    ctx.fillRect(bbox.x1, bbox.y1 - 20, textWidth + 8, 20);

    // Draw label text
    ctx.fillStyle = '#000';
    ctx.fillText(label, bbox.x1 + 4, bbox.y1 - 5);
  });
}

/**
 * Draw pose skeleton on HTML Canvas
 * @param ctx - Canvas 2D context
 * @param people - Array of people with keypoints
 * @param confidenceThreshold - Minimum keypoint confidence to draw (default: 0.3)
 */
export function drawPoseOnCanvas(
  ctx: CanvasRenderingContext2D,
  people: Array<{
    bbox: { x1: number; y1: number; x2: number; y2: number; confidence: number };
    keypoints: Array<{ x: number; y: number; score: number; visible: boolean }>;
  }>,
  confidenceThreshold: number = 0.3
): void {
  // COCO17 skeleton connections (correct MMPose format)
  // Keypoint order: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
  // 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
  // 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
  // 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
  const skeleton = [
    [0, 1], [0, 2], // nose to eyes
    [1, 3], [2, 4], // eyes to ears
    [5, 6], // shoulders
    [5, 7], [7, 9], // left arm
    [6, 8], [8, 10], // right arm
    [5, 11], [6, 12], // shoulders to hips
    [11, 12], // hips
    [11, 13], [13, 15], // left leg
    [12, 14], [14, 16], // right leg
  ];

  const keypointColors = [
    '#FF0000', '#FF0000', '#FF0000', '#FF0000', '#FF0000', // Head
    '#00FF00', '#00FF00', // Shoulders
    '#00FF00', '#00FF00', '#00FF00', // Left arm
    '#00FF00', '#00FF00', '#00FF00', // Right arm
    '#0000FF', '#0000FF', // Torso
    '#0000FF', // Hips
    '#0000FF', '#0000FF', '#0000FF', // Left leg
    '#0000FF', '#0000FF', '#0000FF', // Right leg
  ];

  const skeletonColors = [
    '#FF0000', '#FF0000', '#FF0000', '#FF0000', // Head
    '#00FF00', // Shoulders
    '#00FF00', '#00FF00', // Left arm
    '#00FF00', '#00FF00', // Right arm
    '#0000FF', '#0000FF', // Torso
    '#0000FF', // Hips
    '#0000FF', '#0000FF', '#0000FF', // Left leg
    '#0000FF', '#0000FF', '#0000FF', // Right leg
  ];

  people.forEach((person, personIdx) => {
    const baseColor = `hsl(${personIdx * 60}, 80%, 50%)`;
    const { bbox, keypoints } = person;

    // Draw bounding box
    ctx.strokeStyle = baseColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

    // Draw label
    const label = `Person ${personIdx + 1} ${(bbox.confidence * 100).toFixed(0)}%`;
    ctx.font = 'bold 12px sans-serif';
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = baseColor;
    ctx.fillRect(bbox.x1, bbox.y1 - 20, textWidth + 8, 20);
    ctx.fillStyle = '#000';
    ctx.fillText(label, bbox.x1 + 4, bbox.y1 - 5);

    // Draw skeleton lines
    skeleton.forEach((link, linkIdx) => {
      const [k1, k2] = link;
      const kp1 = keypoints[k1];
      const kp2 = keypoints[k2];

      if (kp1 && kp2 && kp1.visible && kp2.visible) {
        ctx.strokeStyle = skeletonColors[linkIdx] || baseColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(kp1.x, kp1.y);
        ctx.lineTo(kp2.x, kp2.y);
        ctx.stroke();
      }
    });

    // Draw keypoints
    keypoints.forEach((kp, kpIdx) => {
      if (kp.visible) {
        ctx.fillStyle = keypointColors[kpIdx] || baseColor;
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  });
}

/**
 * Draw both detections and pose on canvas (convenience method)
 * @param ctx - Canvas 2D context
 * @param results - Detection or pose results
 * @param mode - 'object' or 'pose'
 */
export function drawResultsOnCanvas(
  ctx: CanvasRenderingContext2D,
  results: any[],
  mode: 'object' | 'pose' = 'object'
): void {
  if (mode === 'object') {
    drawDetectionsOnCanvas(ctx, results);
  } else {
    drawPoseOnCanvas(ctx, results);
  }
}

/**
 * Draw MHR70 mesh-recovery result on a canvas.
 *
 * Draws bbox + 70 keypoint dots + wrist-fan hand skeleton
 * (wrist → third → second → first → tip per finger). Skips endpoints
 * that fall outside the frame — the model predicts all 70 points, even
 * the ones that lie outside the crop, and without filtering they would
 * just stick to the frame edge.
 *
 * @param ctx - Canvas 2D context (already sized to source-frame dimensions).
 * @param person - One `InstantHMRPerson` from the detector result.
 * @param confidenceThreshold - Skip joints below this score. InstantHMR does
 *   not emit per-kpt scores, so this filter is effectively a no-op unless the
 *   caller pre-fills `score` on `keypoints2d`. Default: 0.
 * @param iw - Source frame width (defaults to the canvas width).
 * @param ih - Source frame height (defaults to the canvas height).
 */
export function drawMhr70OnCanvas(
  ctx: CanvasRenderingContext2D,
  person: InstantHMRPerson,
  confidenceThreshold: number = 0,
  iw?: number,
  ih?: number,
): void {
  const width = iw ?? ctx.canvas.width;
  const height = ih ?? ctx.canvas.height;

  const COLOR_LEFT = '#38bdf8';   // sky-400
  const COLOR_RIGHT = '#fb7185';  // rose-400
  const COLOR_CENTER = '#e2e8f0'; // slate-200

  const sideOf: Array<'left' | 'right' | 'center'> = [
    'center', 'left', 'right', 'left', 'right',           // 0..4  face
    'left', 'right',                                      // 5..6  shoulders
    'left', 'right',                                      // 7..8  elbows
    'left', 'right',                                      // 9..10 hips
    'left', 'right',                                      // 11..12 knees
    'left', 'right',                                      // 13..14 ankles
    'left', 'left', 'left',                               // 15..17 left foot
    'right', 'right', 'right',                            // 18..20 right foot
    'right', 'right', 'right', 'right',                   // 21..24 right thumb chain
    'right', 'right', 'right', 'right',                   // 25..28 right index chain
    'right', 'right', 'right', 'right',                   // 29..32 right middle chain
    'right', 'right', 'right', 'right',                   // 33..36 right ring chain
    'right', 'right', 'right', 'right',                   // 37..40 right pinky chain
    'right',                                              // 41 right wrist
    'left', 'left', 'left', 'left',                       // 42..45 left thumb chain
    'left', 'left', 'left', 'left',                       // 46..49 left index chain
    'left', 'left', 'left', 'left',                       // 50..53 left middle chain
    'left', 'left', 'left', 'left',                       // 54..57 left ring chain
    'left', 'left', 'left', 'left',                       // 58..61 left pinky chain
    'left',                                               // 62 left wrist
    'left', 'right',                                      // 63..64 olecranon
    'left', 'right',                                      // 65..66 cubital fossa
    'left', 'right',                                      // 67..68 acromion
    'center',                                             // 69 neck
  ];

  const colorOf = (i: number): string => {
    const s = sideOf[i] ?? 'center';
    return s === 'left' ? COLOR_LEFT : s === 'right' ? COLOR_RIGHT : COLOR_CENTER;
  };

  const { bbox, keypoints2d } = person;

  // Bounding box.
  ctx.strokeStyle = COLOR_CENTER;
  ctx.lineWidth = 2;
  ctx.strokeRect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);

  const label = `Person ${(bbox.confidence * 100).toFixed(0)}%`;
  ctx.font = 'bold 12px sans-serif';
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = COLOR_CENTER;
  ctx.fillRect(bbox.x1, bbox.y1 - 18, tw + 8, 18);
  ctx.fillStyle = '#04202b';
  ctx.fillText(label, bbox.x1 + 4, bbox.y1 - 4);

  // Skeleton edges (wrist-fan hands per SKELETON_EDGES — matches upstream).
  ctx.lineWidth = 2;
  ctx.lineCap = 'round';
  for (const [a, b] of SKELETON_EDGES) {
    const ka = keypoints2d[a];
    const kb = keypoints2d[b];
    if (!ka || !kb) continue;
    // Skip joints outside the frame.
    const aIn = ka.x >= 0 && ka.x < width && ka.y >= 0 && ka.y < height;
    const bIn = kb.x >= 0 && kb.x < width && kb.y >= 0 && kb.y < height;
    if (!aIn || !bIn) continue;
    // Confidence filter (InstantHMR doesn't emit scores; keypoints2d[k].score is undefined).
    const scoreA = ka['score' as keyof typeof ka] as number | undefined;
    const scoreB = kb['score' as keyof typeof kb] as number | undefined;
    if ((scoreA ?? 1) < confidenceThreshold) continue;
    if ((scoreB ?? 1) < confidenceThreshold) continue;

    ctx.strokeStyle = colorOf(a);
    ctx.beginPath();
    ctx.moveTo(ka.x, ka.y);
    ctx.lineTo(kb.x, kb.y);
    ctx.stroke();
  }

  // Keypoint dots.
  for (let i = 0; i < keypoints2d.length; i++) {
    const kp = keypoints2d[i];
    if (!kp) continue;
    if (kp.x < 0 || kp.x >= width || kp.y < 0 || kp.y >= height) continue;
    ctx.fillStyle = colorOf(i);
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * Draw a small info panel (semi-transparent rounded rectangle with one
 * text line per entry) in the top-left corner of a canvas. Used by the
 * demo's video-export overlay to surface the active detector presets
 * and live FPS in the resulting WebM, but generic enough to be reused
 * for any "render-to-video" use case (debug overlay, watermarking,
 * etc.).
 *
 * Sizes scale to canvas width so the panel remains readable at 1080p+
 * native and at smaller previews. The panel sits inside the canvas so
 * `captureStream()` / `MediaRecorder` will encode it like any other
 * pixel — there's no DOM compositing involved.
 *
 * @param ctx    Canvas 2D context (canvas must already be sized).
 * @param iw     Canvas width in pixels.
 * @param ih     Canvas height in pixels (currently unused — kept for API symmetry).
 * @param lines  One entry per row. Strings are drawn verbatim.
 * @param opts   Optional styling: `accentLine` (zero-based index into
 *               `lines` to colour green instead of the default foreground),
 *               `margin` (px from the top-left corner; defaults to
 *               `iw / 120`), and `fontFamily` (defaults to a monospace
 *               stack). All other sizes are derived from canvas width.
 */
export function drawInfoPanel(
  ctx: CanvasRenderingContext2D,
  iw: number,
  ih: number,
  lines: string[],
  opts: { accentLine?: number; margin?: number; fontFamily?: string } = {},
): void {
  if (!lines.length) return;
  void ih; // reserved for future vertical anchoring

  const scale = iw / 1920; // 1.0 at 1080p width, ~0.5 at 540p
  const fontSize = Math.max(18, Math.round(36 * scale));
  const padX = Math.round(16 * scale);
  const padY = Math.round(12 * scale);
  const lineH = Math.round(fontSize * 1.35);
  const cornerR = Math.round(8 * scale);
  const margin = opts.margin ?? Math.round(16 * scale);
  const fontFamily = opts.fontFamily ?? 'ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace';

  ctx.save();
  ctx.font = `600 ${fontSize}px ${fontFamily}`;
  ctx.textBaseline = 'top';

  const widths = lines.map((l) => ctx.measureText(l).width);
  const boxW = Math.max(...widths) + padX * 2;
  const boxH = lineH * lines.length + padY * 2;

  // Rounded-rect background via four arcTo calls. Supported in every
  // modern browser without needing ctx.roundRect.
  const x = margin;
  const y = margin;
  ctx.beginPath();
  ctx.moveTo(x + cornerR, y);
  ctx.arcTo(x + boxW, y, x + boxW, y + boxH, cornerR);
  ctx.arcTo(x + boxW, y + boxH, x, y + boxH, cornerR);
  ctx.arcTo(x, y + boxH, x, y, cornerR);
  ctx.arcTo(x, y, x + boxW, y, cornerR);
  ctx.closePath();
  ctx.fillStyle = 'rgba(11, 12, 16, 0.78)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.55)';
  ctx.lineWidth = Math.max(1, scale);
  ctx.stroke();

  // Foreground colours: accent green for the highlighted line,
  // default neutral foreground for everything else.
  const accent = '#6ee7b7';
  const fg = '#e4e4e7';
  for (let i = 0; i < lines.length; i++) {
    ctx.fillStyle = i === opts.accentLine ? accent : fg;
    ctx.fillText(lines[i], x + padX, y + padY + lineH * i);
  }
  ctx.restore();
}
