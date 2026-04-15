# Hand Gesture Detection Fixes

## Problems Identified

1. **Confidence exceeding 100%** - Pataka Hasta was showing 104% confidence
2. **Poor hand detection** - Hands not being detected reliably
3. **Inaccurate gesture recognition** - Gestures not being identified correctly
4. **Flickering between gestures** - Unstable detection frame-to-frame

## Root Causes & Solutions

### 1. Confidence Calculation Bug (FIXED)
**Problem:** The confidence score could exceed 1.0 (100%) because thumb matches received a 1.2x weight multiplier, and the final score wasn't capped.

**Solution:** 
- Added `min(raw_score, 1.0)` to cap confidence at 100%
- File: `src/hand/hand_detector.py` in `_calculate_match_score()` method

### 2. High Detection Threshold (FIXED)
**Problem:** The `min_detection_confidence` was set to 0.7, which was too strict for various lighting conditions and hand positions.

**Solution:**
- Lowered default `min_detection_confidence` from 0.7 to 0.5
- Updated both `HandGestureDetector.__init__()` and `hand_gesture_demo.py`
- This allows MediaPipe to detect hands more reliably

### 3. Poor Gesture Recognition Accuracy (IMPROVED)
**Problem:** The finger state detection thresholds were too strict, causing misclassification.

**Solutions:**
- **Extended finger threshold:** Lowered from 160° to 150° for better detection of "straight" fingers
- **Folded finger threshold:** Tightened from 110° to 100° for better distinction from "bent" state
- **Tolerance matching:** Added `_has_tolerance_match()` method that allows partial matches between similar states
  - This handles the natural variance in MediaPipe landmark detection
  - Gives 0.3 weight (lower than direct or similar matches)

### 4. Gesture Flickering (FIXED)
**Problem:** Without temporal smoothing, the detected gesture could flicker between frames due to detection noise.

**Solution:**
- Added temporal smoothing with a 5-frame history buffer
- New method `_apply_temporal_smoothing()` stabilizes gesture detection
- Only accepts a new gesture if it appears in the majority of recent frames (≥3 out of 5)
- Averages confidence over the history window for more stable scores
- Added `reset_gesture_history()` method for fresh starts

## Testing the Fixes

Run the hand gesture demo:
```bash
python run_webcam.py
# or
python demos/hand_gesture_demo.py
```

**Expected improvements:**
1. ✅ Confidence never exceeds 100%
2. ✅ Hands detected more reliably (lower threshold)
3. ✅ Better gesture classification accuracy
4. ✅ Stable, non-flickering gesture labels

## Technical Details

### Finger State Detection Hierarchy
1. Check thumb-to-finger touching (overrides other states)
2. Check for 'straight' (angles > 150°)
3. Check for 'slightly_bent' (150-170°)
4. Check for 'curved' (90-150°)
5. Check for 'bent' (110-150°, not folded)
6. Check for 'folded' (curled, < 100° avg)

### Confidence Scoring
- Direct match: 1.0 points (thumb: 1.2x weight)
- Similar states: 0.5 points (thumb: 0.6x)
- Tolerance match: 0.3 points (loose matching)
- Special rules: 1.0 points
- Final score: `min(score, 1.0)` - capped at 100%

### Temporal Smoothing
- Window size: 5 frames
- Minimum stable frames: 3 (majority vote)
- Confidence averaging over history window
- Prevents flickering between similar gestures

## Files Modified

1. `src/hand/hand_detector.py` - Core fixes
2. `demos/hand_gesture_demo.py` - Updated detection threshold

## Recommendations for Better Results

1. **Lighting:** Ensure good, even lighting on hands
2. **Background:** Use a contrasting background (not skin-toned)
3. **Distance:** Keep hands 30-60cm from camera
4. **Speed:** Move hands slowly for stable detection
5. **Angles:** Keep palms facing camera when possible

## Future Improvements (Optional)

If you want even better accuracy, consider:
- Collecting training images for each hasta
- Training a custom ML classifier on top of MediaPipe features
- Adding palm orientation analysis
- Implementing gesture transition detection for dynamic hastas
