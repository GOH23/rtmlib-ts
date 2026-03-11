/**
 * OpenPose18 skeleton configuration
 * 18 keypoints for body pose estimation (OpenPose style)
 */

export const openpose18 = {
  keypoint_info: {
    0: { name: 'nose', id: 0, color: [255, 0, 0] },
    1: { name: 'neck', id: 1, color: [255, 85, 0] },
    2: { name: 'right_shoulder', id: 2, color: [255, 170, 0] },
    3: { name: 'right_elbow', id: 3, color: [255, 255, 0] },
    4: { name: 'right_wrist', id: 4, color: [170, 255, 0] },
    5: { name: 'left_shoulder', id: 5, color: [85, 255, 0] },
    6: { name: 'left_elbow', id: 6, color: [0, 255, 0] },
    7: { name: 'left_wrist', id: 7, color: [0, 255, 85] },
    8: { name: 'right_hip', id: 8, color: [0, 255, 170] },
    9: { name: 'right_knee', id: 9, color: [0, 255, 255] },
    10: { name: 'right_ankle', id: 10, color: [0, 170, 255] },
    11: { name: 'left_hip', id: 11, color: [0, 85, 255] },
    12: { name: 'left_knee', id: 12, color: [0, 0, 255] },
    13: { name: 'left_ankle', id: 13, color: [85, 0, 255] },
    14: { name: 'right_eye', id: 14, color: [170, 0, 255] },
    15: { name: 'left_eye', id: 15, color: [255, 0, 255] },
    16: { name: 'right_ear', id: 16, color: [255, 0, 170] },
    17: { name: 'left_ear', id: 17, color: [255, 0, 85] },
  },
  skeleton_info: {
    0: { link: ['neck', 'right_shoulder'], color: [255, 85, 0] },
    1: { link: ['neck', 'left_shoulder'], color: [255, 85, 0] },
    2: { link: ['right_shoulder', 'right_elbow'], color: [255, 170, 0] },
    3: { link: ['right_elbow', 'right_wrist'], color: [255, 255, 0] },
    4: { link: ['left_shoulder', 'left_elbow'], color: [85, 255, 0] },
    5: { link: ['left_elbow', 'left_wrist'], color: [0, 255, 0] },
    6: { link: ['neck', 'right_hip'], color: [255, 85, 0] },
    7: { link: ['right_hip', 'right_knee'], color: [0, 255, 170] },
    8: { link: ['right_knee', 'right_ankle'], color: [0, 255, 255] },
    9: { link: ['neck', 'left_hip'], color: [255, 85, 0] },
    10: { link: ['left_hip', 'left_knee'], color: [0, 85, 255] },
    11: { link: ['left_knee', 'left_ankle'], color: [0, 0, 255] },
    12: { link: ['nose', 'neck'], color: [255, 0, 0] },
    13: { link: ['nose', 'right_eye'], color: [255, 0, 0] },
    14: { link: ['nose', 'left_eye'], color: [255, 0, 0] },
    15: { link: ['right_eye', 'right_ear'], color: [170, 0, 255] },
    16: { link: ['left_eye', 'left_ear'], color: [255, 0, 255] },
    17: { link: ['right_shoulder', 'left_shoulder'], color: [255, 85, 0] },
    18: { link: ['right_hip', 'left_hip'], color: [255, 85, 0] },
  },
} as const;
