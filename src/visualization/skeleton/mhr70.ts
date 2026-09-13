/**
 * MHR70 skeleton configuration.
 * 70 keypoints for whole-body mesh recovery (InstantHMR model).
 *
 * Colors follow the `jointSide` convention in `core/instanthmrGeometry.ts`:
 *   left side  → sky-400   [56, 189, 248]
 *   right side → rose-400  [251, 113, 133]
 *   centre     → slate-200 [226, 232, 240]
 */

import {
  JOINT_NAMES,
  SKELETON_EDGES,
  jointSide,
} from '../../core/instanthmrGeometry';

const COLOR_LEFT: [number, number, number] = [56, 189, 248];
const COLOR_RIGHT: [number, number, number] = [251, 113, 133];
const COLOR_CENTER: [number, number, number] = [226, 232, 240];

function colorForJoint(id: number): [number, number, number] {
  switch (jointSide(id)) {
    case 'left':
      return COLOR_LEFT;
    case 'right':
      return COLOR_RIGHT;
    default:
      return COLOR_CENTER;
  }
}

export const mhr70 = {
  keypoint_info: Object.fromEntries(
    JOINT_NAMES.map((name, id) => [id, { name, id, color: colorForJoint(id) }]),
  ),
  skeleton_info: Object.fromEntries(
    SKELETON_EDGES.map(([a, b], id) => {
      // Colour the link by the side of its first endpoint — both ends of any
      // edge in MHR70 share a side, so this is unambiguous.
      const side = jointSide(a);
      const color =
        side === 'left' ? COLOR_LEFT :
        side === 'right' ? COLOR_RIGHT :
        COLOR_CENTER;
      return [id, {
        link: [JOINT_NAMES[a] as string, JOINT_NAMES[b] as string],
        color,
      }];
    }),
  ),
} as const;
