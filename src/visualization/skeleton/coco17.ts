/**
 * COCO17 skeleton configuration
 * 17 keypoints for body pose estimation
 */

export const coco17 = {
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
} as const;
