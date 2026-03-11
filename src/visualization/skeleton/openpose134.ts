/**
 * OpenPose134 skeleton configuration
 * 134 keypoints for wholebody pose estimation (OpenPose style)
 */

export const openpose134 = {
  keypoint_info: {
    // Body (0-17)
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
    // Face (18-87) - 70 points
    ...Object.fromEntries(
      Array.from({ length: 70 }, (_, i) => [
        i + 18,
        { name: `face_${i}`, id: i + 18, color: [255, 255, 255] }
      ])
    ),
    // Left hand (88-108) - 21 points
    ...Object.fromEntries(
      Array.from({ length: 21 }, (_, i) => [
        i + 88,
        { name: `left_hand_${i}`, id: i + 88, color: [255, 128, 0] }
      ])
    ),
    // Right hand (109-129) - 21 points
    ...Object.fromEntries(
      Array.from({ length: 21 }, (_, i) => [
        i + 109,
        { name: `right_hand_${i}`, id: i + 109, color: [0, 128, 255] }
      ])
    ),
    // Left foot (130-133) - 4 points
    130: { name: 'left_big_toe', id: 130, color: [0, 255, 128] },
    131: { name: 'left_small_toe', id: 131, color: [0, 255, 128] },
    132: { name: 'left_heel', id: 132, color: [0, 255, 128] },
    133: { name: 'left_foot_center', id: 133, color: [0, 255, 128] },
  },
  skeleton_info: {
    // Body skeleton
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
    // Face skeleton (simplified contour)
    ...Object.fromEntries(
      Array.from({ length: 17 }, (_, i) => [
        i + 19,
        { link: [`face_${i}`, `face_${(i + 1) % 17}`], color: [255, 255, 255] }
      ])
    ),
    // Left hand skeleton
    36: { link: ['left_hand_0', 'left_hand_1'], color: [255, 128, 0] },
    37: { link: ['left_hand_1', 'left_hand_2'], color: [255, 128, 0] },
    38: { link: ['left_hand_2', 'left_hand_3'], color: [255, 128, 0] },
    39: { link: ['left_hand_3', 'left_hand_4'], color: [255, 128, 0] },
    40: { link: ['left_hand_0', 'left_hand_5'], color: [255, 128, 0] },
    41: { link: ['left_hand_5', 'left_hand_6'], color: [255, 128, 0] },
    42: { link: ['left_hand_6', 'left_hand_7'], color: [255, 128, 0] },
    43: { link: ['left_hand_7', 'left_hand_8'], color: [255, 128, 0] },
    44: { link: ['left_hand_0', 'left_hand_9'], color: [255, 128, 0] },
    45: { link: ['left_hand_9', 'left_hand_10'], color: [255, 128, 0] },
    46: { link: ['left_hand_10', 'left_hand_11'], color: [255, 128, 0] },
    47: { link: ['left_hand_11', 'left_hand_12'], color: [255, 128, 0] },
    48: { link: ['left_hand_0', 'left_hand_13'], color: [255, 128, 0] },
    49: { link: ['left_hand_13', 'left_hand_14'], color: [255, 128, 0] },
    50: { link: ['left_hand_14', 'left_hand_15'], color: [255, 128, 0] },
    51: { link: ['left_hand_15', 'left_hand_16'], color: [255, 128, 0] },
    52: { link: ['left_hand_0', 'left_hand_17'], color: [255, 128, 0] },
    53: { link: ['left_hand_17', 'left_hand_18'], color: [255, 128, 0] },
    54: { link: ['left_hand_18', 'left_hand_19'], color: [255, 128, 0] },
    55: { link: ['left_hand_19', 'left_hand_20'], color: [255, 128, 0] },
    // Right hand skeleton
    56: { link: ['right_hand_0', 'right_hand_1'], color: [0, 128, 255] },
    57: { link: ['right_hand_1', 'right_hand_2'], color: [0, 128, 255] },
    58: { link: ['right_hand_2', 'right_hand_3'], color: [0, 128, 255] },
    59: { link: ['right_hand_3', 'right_hand_4'], color: [0, 128, 255] },
    60: { link: ['right_hand_0', 'right_hand_5'], color: [0, 128, 255] },
    61: { link: ['right_hand_5', 'right_hand_6'], color: [0, 128, 255] },
    62: { link: ['right_hand_6', 'right_hand_7'], color: [0, 128, 255] },
    63: { link: ['right_hand_7', 'right_hand_8'], color: [0, 128, 255] },
    64: { link: ['right_hand_0', 'right_hand_9'], color: [0, 128, 255] },
    65: { link: ['right_hand_9', 'right_hand_10'], color: [0, 128, 255] },
    66: { link: ['right_hand_10', 'right_hand_11'], color: [0, 128, 255] },
    67: { link: ['right_hand_11', 'right_hand_12'], color: [0, 128, 255] },
    68: { link: ['right_hand_0', 'right_hand_13'], color: [0, 128, 255] },
    69: { link: ['right_hand_13', 'right_hand_14'], color: [0, 128, 255] },
    70: { link: ['right_hand_14', 'right_hand_15'], color: [0, 128, 255] },
    71: { link: ['right_hand_15', 'right_hand_16'], color: [0, 128, 255] },
    72: { link: ['right_hand_0', 'right_hand_17'], color: [0, 128, 255] },
    73: { link: ['right_hand_17', 'right_hand_18'], color: [0, 128, 255] },
    74: { link: ['right_hand_18', 'right_hand_19'], color: [0, 128, 255] },
    75: { link: ['right_hand_19', 'right_hand_20'], color: [0, 128, 255] },
  },
} as const;
