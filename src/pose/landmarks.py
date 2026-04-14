"""
MediaPipe Pose Landmark Reference
==================================
MediaPipe Pose detects 33 landmarks on the human body.

INDEX | LANDMARK NAME       | DESCRIPTION
------|---------------------|----------------------------------
0     | nose                | Tip of the nose
1     | left_eye_inner      | Inner corner of left eye
2     | left_eye            | Center of left eye
3     | left_eye_outer      | Outer corner of left eye
4     | right_eye_inner     | Inner corner of right eye
5     | right_eye           | Center of right eye
6     | right_eye_outer     | Outer corner of right eye
7     | left_ear            | Left ear
8     | right_ear           | Right ear
9     | mouth_left          | Left corner of mouth
10    | mouth_right         | Right corner of mouth
11    | left_shoulder       | Left shoulder joint
12    | right_shoulder      | Right shoulder joint
13    | left_elbow          | Left elbow joint
14    | right_elbow         | Right elbow joint
15    | left_wrist          | Left wrist joint
16    | right_wrist         | Right wrist joint
17    | left_pinky          | Left pinky finger (MCP joint)
18    | right_pinky         | Right pinky finger (MCP joint)
19    | left_index          | Left index finger (MCP joint)
20    | right_index         | Right index finger (MCP joint)
21    | left_thumb          | Left thumb (IP joint)
22    | right_thumb         | Right thumb (IP joint)
23    | left_hip            | Left hip joint
24    | right_hip           | Right hip joint
25    | left_knee           | Left knee joint
26    | right_knee          | Right knee joint
27    | left_ankle          | Left ankle joint
28    | right_ankle         | Right ankle joint
29    | left_heel           | Left heel
30    | right_heel          | Right heel
31    | left_foot_index     | Left foot index (toe)
32    | right_foot_index    | Right foot index (toe)

Body Connections (POSE_CONNECTIONS):
- Face: nose, eyes, ears, mouth corners
- Upper Body: shoulders, elbows, wrists, fingers
- Lower Body: hips, knees, ankles, heels, feet
- Torso: left/right shoulder to hip connections
"""

# Landmark index constants for easy reference
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32
