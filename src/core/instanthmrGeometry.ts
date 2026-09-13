// Pure geometry and constants for the InstantHMR 3D-pose model.
//
// Verbatim port of `instanthmr/inference.py` (class InstantHMR) and
// `instanthmr/skeleton.py`. No DOM, no ORT, no library dependencies — this
// module is pure math and must verify against the Python reference outside
// the browser.
//
// Any discrepancy in preprocessing breaks the prediction silently: the model
// still returns 70 joints — just not the right ones. The comments below point
// to the Python reference line that each formula was checked against.

// --------------------------------------------------------------------------
// Constants (inference.py:35-38)
// --------------------------------------------------------------------------

export const INPUT_SIZE = 224;
/** Square crop expansion around the detector bbox, as used during training. */
export const CROP_EXPAND = 1.2;
export const NUM_JOINTS = 70;

export const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
export const IMAGENET_STD = [0.229, 0.224, 0.225] as const;

// --------------------------------------------------------------------------
// Types
// --------------------------------------------------------------------------

/** Bounding box from a person detector (top-left + dimensions). */
export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Square crop actually fed to the model.
 * Required to back-project joints_2d into the source frame: it cannot be
 * recovered from the raw detector bbox, since the crop is 1.2× larger.
 */
export interface CropBox {
  x0: number;
  y0: number;
  size: number;
}

// --------------------------------------------------------------------------
// Preprocessing (port of InstantHMR._preprocess, inference.py:385-436)
// --------------------------------------------------------------------------

/** Square crop 1.2× around the bbox centre (inference.py:404-407). */
export function cropBoxFor(bbox: BBox): CropBox {
  const cx = bbox.x + bbox.w / 2;
  const cy = bbox.y + bbox.h / 2;
  const size = Math.max(bbox.w, bbox.h) * CROP_EXPAND;
  return { x0: cx - size / 2, y0: cy - size / 2, size };
}

/**
 * CLIFF conditioning (inference.py:398-402).
 *
 * Computed against the RAW detector bbox and the FULL frame size — not the
 * expanded crop, and not the magic 200 constant from the original CLIFF paper.
 * The third component is the fraction of the frame the person occupies; it
 * sets the scale for the model and directly determines cam_trans.
 */
export function cliffCondFor(bbox: BBox, iw: number, ih: number): Float32Array {
  const cx = bbox.x + bbox.w / 2;
  const cy = bbox.y + bbox.h / 2;
  return new Float32Array([
    2 * (cx / iw) - 1,
    2 * (cy / ih) - 1,
    Math.max(bbox.w, bbox.h) / Math.max(iw, ih),
  ]);
}

/**
 * RGBA pixels → NCHW float32 with ImageNet normalization (inference.py:431-434).
 *
 * Writes one frame at `offset`; the caller allocates the full output buffer
 * for batched inference (`n * 3 * INPUT_SIZE^2`) and passes `i * 3 * plane`
 * as the offset for person `i`. This avoids per-person allocations.
 */
export function pixelsToNCHW(
  rgba: Uint8ClampedArray,
  out: Float32Array,
  offset: number,
): void {
  const plane = INPUT_SIZE * INPUT_SIZE;
  const inv255 = 1 / 255;
  for (let i = 0, p = 0; p < plane; p++, i += 4) {
    out[offset + p] = (rgba[i] * inv255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    out[offset + plane + p] = (rgba[i + 1] * inv255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    out[offset + 2 * plane + p] = (rgba[i + 2] * inv255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
}

// --------------------------------------------------------------------------
// Postprocessing
// --------------------------------------------------------------------------

/**
 * joints_2d → source-frame pixels (inference.py:242-246).
 *
 * The model emits normalized crop coordinates in **[-1, 1]** (HF README:
 * "joints_2d (N, 70, 2) keypoints in normalised crop coords [-1, 1]"), so
 * first (v + 1) / 2, then scale by crop.size.
 */
export function denormalizeJoints2D(
  joints2dNorm: Float32Array,
  crop: CropBox,
): Float32Array {
  const out = new Float32Array(joints2dNorm.length);
  for (let i = 0; i < joints2dNorm.length; i += 2) {
    out[i] = crop.x0 + (joints2dNorm[i] + 1) * 0.5 * crop.size;
    out[i + 1] = crop.y0 + (joints2dNorm[i + 1] + 1) * 0.5 * crop.size;
  }
  return out;
}

/** Unpacks half-float bits to float32, for fp16 model exports. */
export function float16ToFloat32(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

// --------------------------------------------------------------------------
// MHR70 skeleton — exact port of instanthmr/skeleton.py
// --------------------------------------------------------------------------

/** 70 joint names in MHR70 order (skeleton.py:11-46). */
export const JOINT_NAMES: readonly string[] = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',                     // 0-4
  'left_shoulder', 'right_shoulder',                                            // 5-6
  'left_elbow', 'right_elbow',                                                  // 7-8
  'left_hip', 'right_hip',                                                      // 9-10
  'left_knee', 'right_knee',                                                    // 11-12
  'left_ankle', 'right_ankle',                                                  // 13-14
  'left_big_toe_tip', 'left_small_toe_tip', 'left_heel',                        // 15-17
  'right_big_toe_tip', 'right_small_toe_tip', 'right_heel',                     // 18-20
  'right_thumb_tip', 'right_thumb_first_joint',
  'right_thumb_second_joint', 'right_thumb_third_joint',                        // 21-24
  'right_index_tip', 'right_index_first_joint',
  'right_index_second_joint', 'right_index_third_joint',                        // 25-28
  'right_middle_tip', 'right_middle_first_joint',
  'right_middle_second_joint', 'right_middle_third_joint',                      // 29-32
  'right_ring_tip', 'right_ring_first_joint',
  'right_ring_second_joint', 'right_ring_third_joint',                          // 33-36
  'right_pinky_tip', 'right_pinky_first_joint',
  'right_pinky_second_joint', 'right_pinky_third_joint',                        // 37-40
  'right_wrist',                                                                // 41
  'left_thumb_tip', 'left_thumb_first_joint',
  'left_thumb_second_joint', 'left_thumb_third_joint',                          // 42-45
  'left_index_tip', 'left_index_first_joint',
  'left_index_second_joint', 'left_index_third_joint',                          // 46-49
  'left_middle_tip', 'left_middle_first_joint',
  'left_middle_second_joint', 'left_middle_third_joint',                        // 50-53
  'left_ring_tip', 'left_ring_first_joint',
  'left_ring_second_joint', 'left_ring_third_joint',                            // 54-57
  'left_pinky_tip', 'left_pinky_first_joint',
  'left_pinky_second_joint', 'left_pinky_third_joint',                          // 58-61
  'left_wrist',                                                                 // 62
  'left_olecranon', 'right_olecranon',                                          // 63-64
  'left_cubital_fossa', 'right_cubital_fossa',                                  // 65-66
  'left_acromion', 'right_acromion',                                            // 67-68
  'neck',                                                                       // 69
];

/**
 * MHR70 skeleton edges.
 *
 * Mirrors `instanthmr/skeleton.py:49-65`: face / head-neck-shoulders /
 * torso / arms / legs / feet as before, but hands are drawn as a wrist-fan
 * (5 straight lines from each wrist to the 5 finger tips) rather than as
 * per-phalanx chains. The fan matches what `instanthmr/visualizer.py:280`
 * draws on the upstream demo; per-phalanx traversal would skip the middle
 * joints visually.
 */
const RIGHT_WRIST = 41;
const LEFT_WRIST = 62;

export const SKELETON_EDGES: ReadonlyArray<readonly [number, number]> = [
  // Face
  [0, 1], [0, 2], [1, 2], [1, 3], [2, 4],
  // Head ↔ neck ↔ shoulders
  [0, 69], [69, 5], [69, 6],
  // Torso
  [5, 6], [5, 9], [6, 10], [9, 10],
  // Arms
  [5, 7], [7, LEFT_WRIST],
  [6, 8], [8, RIGHT_WRIST],
  // Legs
  [9, 11], [11, 13],
  [10, 12], [12, 14],
  // Feet: ankle → big toe, small toe, heel
  [13, 15], [13, 16], [13, 17],
  [14, 18], [14, 19], [14, 20],
  // Hands — wrist-fan: one edge per finger, wrist → tip.
  // Thumb/index/middle/ring/pinky tips at 21/25/29/33/37 (right) and
  // 42/46/50/54/58 (left).
  [RIGHT_WRIST, 21], [RIGHT_WRIST, 25],
  [RIGHT_WRIST, 29], [RIGHT_WRIST, 33], [RIGHT_WRIST, 37],
  [LEFT_WRIST, 42], [LEFT_WRIST, 46],
  [LEFT_WRIST, 50], [LEFT_WRIST, 54], [LEFT_WRIST, 58],
];

export type Side = 'left' | 'right' | 'center';

/** Body side by joint index — used to color the skeleton. */
export function jointSide(index: number): Side {
  const name = JOINT_NAMES[index] ?? '';
  if (name.startsWith('left_')) return 'left';
  if (name.startsWith('right_')) return 'right';
  return 'center';
}
