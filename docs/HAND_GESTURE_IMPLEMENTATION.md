# Bharathanatyam Hand Gesture Recognition System

## Overview

This implementation provides a comprehensive hand gesture recognition system for **Bharathanatyam classical dance**, capable of detecting and classifying **52 traditional hastas (hand gestures)** from the Bharathanatyam dataset.

### What's Included

- **28 Asamyuta Hastas** - Single hand gestures
- **24 Samyuta Hastas** - Combined two-hand gestures
- Real-time detection using MediaPipe hand tracking
- Finger state analysis and gesture classification
- Visualization and demo scripts

---

## Implementation Architecture

### Process Flow

```
Webcam Input
    ↓
MediaPipe Hand Detection (21 landmarks per hand)
    ↓
Finger State Analysis
    ↓
Gesture Classification (matching against dataset rules)
    ↓
Visualization & Output
```

---

## File Structure

### Core Modules

#### 1. `src/hand/gesture_definitions.py`
**Purpose**: Loads and manages gesture definitions from JSON dataset files

**Key Components**:
- `BharathanatyamGestures` class - Central manager for all gesture data
- Loads `asamyuta_hastas.json` (single hand gestures)
- Loads `samyuta_hastas.json` (combined hand gestures)
- Provides utilities to query gestures by name, index, or type
- Contains MediaPipe hand landmark constants (21 points)

**MediaPipe Hand Landmarks**:
```
0: Wrist
1-4: Thumb (CMC, MCP, IP, TIP)
5-8: Index finger (MCP, PIP, DIP, TIP)
9-12: Middle finger (MCP, PIP, DIP, TIP)
13-16: Ring finger (MCP, PIP, DIP, TIP)
17-20: Pinky finger (MCP, PIP, DIP, TIP)
```

#### 2. `src/hand/hand_detector.py`
**Purpose**: Real-time hand detection and gesture classification

**Key Components**:
- `HandGestureDetector` class - Main detection and classification engine
- MediaPipe Hands integration for hand landmark detection
- Finger state analysis algorithms
- Gesture matching engine with confidence scoring

**Key Methods**:
- `find_hands(img)` - Detect hand landmarks in image
- `analyze_finger_states(landmarks)` - Determine state of each finger
- `classify_gesture(landmarks)` - Match finger states to gesture definitions
- `detect_gestures(img)` - Full pipeline: detect + classify

---

## How Gesture Recognition Works

### Step 1: Hand Landmark Detection

The system uses **MediaPipe Hands** to detect 21 landmarks on each hand in real-time from video frames. These landmarks represent key joints:
- Wrist base
- Finger joints (MCP, PIP, DIP) for each finger
- Finger tips

### Step 2: Finger State Analysis

For each detected hand, the system analyzes the state of each finger:

**Finger States**:
- `straight` - Finger is fully extended
- `bent` - Finger is partially bent
- `folded` - Finger is curled into palm
- `touching_<finger>` - Finger tip is touching another finger
- `curved` - Finger has a slight curve
- `spread` - Fingers are spread apart (V-shape)

**Detection Logic**:
```python
# Example: Check if index finger is extended
def _is_finger_extended(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx):
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]
    dip = landmarks[dip_idx]
    tip = landmarks[tip_idx]
    
    # Finger is extended if joints stack upward
    return (pip.y < mcp.y and dip.y < pip.y and tip.y < dip.y)
```

### Step 3: Gesture Classification

The system compares detected finger states against the **detection rules** defined in the JSON dataset:

**Example - Pataka Hasta**:
```json
{
  "name": "Pataka Hasta",
  "detection_rules": {
    "thumb": "straight",
    "index": "straight",
    "middle": "straight",
    "ring": "straight",
    "pinky": "straight"
  }
}
```

**Matching Algorithm**:
1. Analyze current finger states from detected landmarks
2. Compare against each gesture's detection rules
3. Calculate match score (0.0 to 1.0)
4. Return gesture with highest confidence (if > 0.5 threshold)

**Confidence Scoring**:
- Direct state match: 1.0 points
- Partial/similar state match: 0.5 points
- Special rules (V-shape, C-curve, etc.): Checked separately

### Step 4: Visualization

Detected gestures are displayed with:
- Gesture name label
- Confidence score
- Color-coded by confidence level:
  - 🟢 Green: >80% confidence
  - 🟡 Yellow: 60-80% confidence
  - 🔴 Red: <60% confidence

---

## Dataset Details

### Asamyuta Hastas (Single Hand Gestures)

These are gestures performed with one hand:

1. **Pataka Hasta** - All fingers straight and together
2. **Tripataka Hasta** - Ring finger bent
3. **Ardhapataka Hasta** - Ring and pinky bent
4. **Kartareemukha Hasta** - Index and middle in V-shape
5. **Mayura Hasta** - Peacock gesture
6. **Ardhachandra Hasta** - C-shaped curve
7. **Arala Hasta** - Index finger bent
8. **Shukatunda Hasta** - Index and ring bent
9. **Mushti Hasta** - Fist gesture
10. **Shikhara Hasta** - Thumb extended from fist
11. **Kapitha Hasta** - Index touching thumb
12. **Katakamukha Hasta** - Three fingers touching
13. **Suchi Hasta** - Index finger pointed
14. **Chandrakala Hasta** - L-shape with thumb and index
15. **Padmakosha Hasta** - All fingers curved (grasping ball)
16. **Sarpasheersha Hasta** - Cobra hood shape
17. **Mrugasheersha Hasta** - Deer head shape
18. **Simhamukha Hasta** - Lion face
19. **Langoola Hasta** - Tail gesture
20. **Alapadma Hasta** - Lotus flower (fingers spread)
21. **Chatura Hasta** - Graceful gesture
22. **Bhramara Hasta** - Bee gesture
23. **Hamsasya Hasta** - Swan beak
24. **Hamsapaksha Hasta** - Swan wing
25. **Samdamsha Hasta** - Dynamic opening/closing
26. **Mukula Hasta** - All fingertips touching
27. **Tamrachooda Hasta** - Rooster gesture
28. **Trishoola Hasta** - Trident (three fingers)

### Samyuta Hastas (Combined Hand Gestures)

These require both hands working together:

1. **Anjali Hasta** - Palms pressed together (prayer)
2. **Kapota Hasta** - Hollow cupped hands
3. **Karkata Hasta** - Interlocked fingers
4. **Swastika Hasta** - Crossed hands (X shape)
5. **Dola Hasta** - Arms stretched at hips
6. **Pushpaputa Hasta** - Bowl receiving gesture
7. **Utsanga Hasta** - Arms crossed on shoulders
8. **Shivalinga Hasta** - Representation of Shiva
9. **Katakavardhana Hasta** - Crossed Katakamukha
10. **Kartareeswastika Hasta** - Crossed Kartareemukha
11. **Shakata Hasta** - Demon gesture
12. **Shankha Hasta** - Conch shell
13. **Chakra Hasta** - Wheel/disc shape
14. **Samputa Hasta** - Closed box
15. **Paasha Hasta** - Chain/rope
16. **Keelaka Hasta** - Linked pinkies
17. **Matsya Hasta** - Fish
18. **Koorma Hasta** - Tortoise
19. **Varaha Hasta** - Boar
20. **Garuda Hasta** - Eagle wings
21. **Nagabandha Hasta** - Snake bond
22. **Khatva Hasta** - Bed/cot
23. **Bherundha Hasta** - Bird couple
24. **Avahittha Hasta** - Love gesture

---

## Usage

### Running the Demo

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows

# Run the hand gesture detection demo
python demos/hand_gesture_demo.py
```

### Keyboard Controls

- **q** or **ESC** - Quit the demo
- **i** - Toggle detailed finger state information
- **s** - Save screenshot of current frame

### Programmatic Usage

```python
from src.hand import HandGestureDetector

# Initialize detector
detector = HandGestureDetector(max_num_hands=2)

# Process a frame
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Detect gestures
results = detector.detect_gestures(frame, draw=True)

# Access results
for result in results:
    print(f"Gesture: {result['gesture']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Finger States: {result['finger_states']}")
```

---

## Technical Implementation Details

### Finger State Detection Algorithms

#### 1. **Extended Finger Detection**
A finger is considered "straight/extended" when its joints form a vertical line ascending from the palm:
```
TIP
 │
DIP
 │
PIP
 │
MCP
```

#### 2. **Folded Finger Detection**
A finger is "folded" when the tip drops below the PIP joint (toward the palm).

#### 3. **Bent Finger Detection**
A finger is "bent" when it's neither fully extended nor fully folded.

#### 4. **Touching Detection**
Calculates Euclidean distance between finger tips in 3D space (x, y, z). If distance < threshold (0.05 normalized units), fingers are considered touching.

#### 5. **Special Pattern Recognition**
- **V-shape**: Large distance between index and middle finger tips
- **C-curve**: Thumb and index form semicircle
- **Cobra hood**: All fingers slightly bent and grouped
- **All fingertips touching**: Multiple touching relationships detected

### Confidence Scoring System

The classification uses a **rule-based matching** approach:

```python
match_score = matched_rules / total_rules
```

- **100% match**: All finger states match perfectly
- **75-99%**: Most rules match with some partial matches
- **50-74%**: Moderate match (may still be classified)
- **<50%**: Unknown gesture

### Handling Variations

The system accounts for:
- Different hand sizes (normalized coordinates)
- Hand orientation (left vs. right hand detection)
- Partial occlusion (visibility thresholds)
- Similar gesture states (fuzzy matching)

---

## Limitations & Future Improvements

### Current Limitations

1. **2D Analysis**: Uses primarily 2D coordinates; depth (z-axis) is less reliable
2. **Lighting Dependent**: MediaPipe performance varies with lighting conditions
3. **Hand Speed**: Fast movements may cause detection drops
4. **Similar Gestures**: Some gestures with subtle differences may be confused
5. **Samyuta Gestures**: Combined hand gestures require more complex spatial relationship detection (not fully implemented)

### Potential Enhancements

1. **Temporal Analysis**: Track gestures across multiple frames for stability
2. **Machine Learning**: Train a classifier on labeled gesture data
3. **3D Hand Reconstruction**: Use depth information for better accuracy
4. **Dynamic Gesture Recognition**: Detect gesture transitions and sequences
5. **Cultural Context**: Add detailed meaning explanations and usage examples
6. **Audio Feedback**: Voice announcements of detected gestures
7. **Recording & Playback**: Save gesture sessions for review

---

## Integration with Multimodal System

This hand gesture module integrates with the broader multimodal analysis system:

- **Pose Detection** (`src/pose/`): Body posture analysis
- **Face Detection** (`src/face/`): Facial expression recognition
- **Hand Gestures** (`src/hand/`): Bharathanatyam hasta detection

Combined, these provide comprehensive dance movement analysis for Bharathanatyam performance evaluation.

---

## Troubleshooting

### Issue: No hands detected
- **Solution**: Ensure good lighting, hands clearly visible, not too close/far from camera

### Issue: Low confidence scores
- **Solution**: Practice gestures clearly, hold poses steady for 1-2 seconds

### Issue: Wrong gesture classified
- **Solution**: Check finger positions match detection rules exactly; adjust hand angle

### Issue: Webcam not opening
- **Solution**: Check camera permissions, close other apps using the camera

---

## References

- **Dataset**: Bharathanatyam Hastas JSON files (provided)
- **MediaPipe**: https://google.github.io/mediapipe/solutions/hands.html
- **Bharathanatyam**: Classical Indian dance form from Tamil Nadu

---

## Author

Implementation for multimodal analysis system - NIDHISHREE N Internship Project
