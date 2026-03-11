/**
 * Drawing utilities for visualization
 * Based on rtmlib Python library
 */

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
  // COCO17 skeleton connections
  const skeleton = [
    [0, 1], [0, 2], [1, 3], [2, 4], // Head
    [5, 6], // Shoulders
    [5, 7], [7, 9], // Left arm
    [6, 8], [8, 10], // Right arm
    [5, 11], [6, 12], // Torso
    [11, 12], // Hips
    [11, 13], [13, 15], // Left leg
    [12, 14], [14, 16], // Right leg
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
