/**
 * COCO133 skeleton configuration
 * 133 keypoints for wholebody pose estimation (body + hands + face)
 */

export const coco133 = {
  keypoint_info: {
    // Body (0-16)
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
    // Face (17-84) - 68 points
    ...Object.fromEntries(
      Array.from({ length: 68 }, (_, i) => [
        i + 17,
        { name: `face_${i}`, id: i + 17, color: [255, 255, 255] }
      ])
    ),
    // Left hand (85-105) - 21 points
    ...Object.fromEntries(
      Array.from({ length: 21 }, (_, i) => [
        i + 85,
        { name: `left_hand_${i}`, id: i + 85, color: [255, 128, 0] }
      ])
    ),
    // Right hand (106-126) - 21 points
    ...Object.fromEntries(
      Array.from({ length: 21 }, (_, i) => [
        i + 106,
        { name: `right_hand_${i}`, id: i + 106, color: [0, 128, 255] }
      ])
    ),
    // Left foot (127-130) - 4 points
    ...Object.fromEntries(
      Array.from({ length: 4 }, (_, i) => [
        i + 127,
        { name: `left_foot_${i}`, id: i + 127, color: [0, 255, 128] }
      ])
    ),
    // Right foot (131-132) - 2 points
    131: { name: 'right_foot_0', id: 131, color: [128, 0, 255] },
    132: { name: 'right_foot_1', id: 132, color: [128, 0, 255] },
  },
  skeleton_info: {
    // Body skeleton
    0: { link: ['left_ankle', 'left_knee'], color: [255, 51, 255] },
    1: { link: ['left_knee', 'left_hip'], color: [255, 51, 255] },
    2: { link: ['left_hip', 'right_hip'], color: [255, 255, 51] },
    3: { link: ['right_hip', 'right_knee'], color: [255, 51, 255] },
    4: { link: ['right_knee', 'right_ankle'], color: [255, 51, 255] },
    5: { link: ['left_hip', 'left_shoulder'], color: [255, 255, 51] },
    6: { link: ['left_shoulder', 'left_elbow'], color: [255, 255, 51] },
    7: { link: ['left_elbow', 'left_wrist'], color: [255, 255, 51] },
    8: { link: ['right_hip', 'right_shoulder'], color: [255, 255, 51] },
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
    // Face skeleton (simplified)
    ...Object.fromEntries(
      Array.from({ length: 17 }, (_, i) => [
        i + 21,
        { link: [`face_${i}`, `face_${(i + 1) % 17}`], color: [255, 255, 255] }
      ])
    ),
    // Left hand skeleton
    38: { link: ['left_hand_0', 'left_hand_1'], color: [255, 128, 0] },
    39: { link: ['left_hand_1', 'left_hand_2'], color: [255, 128, 0] },
    40: { link: ['left_hand_2', 'left_hand_3'], color: [255, 128, 0] },
    41: { link: ['left_hand_3', 'left_hand_4'], color: [255, 128, 0] },
    42: { link: ['left_hand_0', 'left_hand_5'], color: [255, 128, 0] },
    43: { link: ['left_hand_5', 'left_hand_6'], color: [255, 128, 0] },
    44: { link: ['left_hand_6', 'left_hand_7'], color: [255, 128, 0] },
    45: { link: ['left_hand_7', 'left_hand_8'], color: [255, 128, 0] },
    46: { link: ['left_hand_0', 'left_hand_9'], color: [255, 128, 0] },
    47: { link: ['left_hand_9', 'left_hand_10'], color: [255, 128, 0] },
    48: { link: ['left_hand_10', 'left_hand_11'], color: [255, 128, 0] },
    49: { link: ['left_hand_11', 'left_hand_12'], color: [255, 128, 0] },
    50: { link: ['left_hand_0', 'left_hand_13'], color: [255, 128, 0] },
    51: { link: ['left_hand_13', 'left_hand_14'], color: [255, 128, 0] },
    52: { link: ['left_hand_14', 'left_hand_15'], color: [255, 128, 0] },
    53: { link: ['left_hand_15', 'left_hand_16'], color: [255, 128, 0] },
    54: { link: ['left_hand_0', 'left_hand_17'], color: [255, 128, 0] },
    55: { link: ['left_hand_17', 'left_hand_18'], color: [255, 128, 0] },
    56: { link: ['left_hand_18', 'left_hand_19'], color: [255, 128, 0] },
    57: { link: ['left_hand_19', 'left_hand_20'], color: [255, 128, 0] },
    // Right hand skeleton
    58: { link: ['right_hand_0', 'right_hand_1'], color: [0, 128, 255] },
    59: { link: ['right_hand_1', 'right_hand_2'], color: [0, 128, 255] },
    60: { link: ['right_hand_2', 'right_hand_3'], color: [0, 128, 255] },
    61: { link: ['right_hand_3', 'right_hand_4'], color: [0, 128, 255] },
    62: { link: ['right_hand_0', 'right_hand_5'], color: [0, 128, 255] },
    63: { link: ['right_hand_5', 'right_hand_6'], color: [0, 128, 255] },
    64: { link: ['right_hand_6', 'right_hand_7'], color: [0, 128, 255] },
    65: { link: ['right_hand_7', 'right_hand_8'], color: [0, 128, 255] },
    66: { link: ['right_hand_0', 'right_hand_9'], color: [0, 128, 255] },
    67: { link: ['right_hand_9', 'right_hand_10'], color: [0, 128, 255] },
    68: { link: ['right_hand_10', 'right_hand_11'], color: [0, 128, 255] },
    69: { link: ['right_hand_11', 'right_hand_12'], color: [0, 128, 255] },
    70: { link: ['right_hand_0', 'right_hand_13'], color: [0, 128, 255] },
    71: { link: ['right_hand_13', 'right_hand_14'], color: [0, 128, 255] },
    72: { link: ['right_hand_14', 'right_hand_15'], color: [0, 128, 255] },
    73: { link: ['right_hand_15', 'right_hand_16'], color: [0, 128, 255] },
    74: { link: ['right_hand_0', 'right_hand_17'], color: [0, 128, 255] },
    75: { link: ['right_hand_17', 'right_hand_18'], color: [0, 128, 255] },
    76: { link: ['right_hand_18', 'right_hand_19'], color: [0, 128, 255] },
    77: { link: ['right_hand_19', 'right_hand_20'], color: [0, 128, 255] },
  },
} as const;
