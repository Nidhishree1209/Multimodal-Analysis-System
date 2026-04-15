"""
Bharathanatyam Hand Gesture Detection Demo
===========================================
This script demonstrates real-time detection of Bharathanatyam hand gestures
(hastas) using MediaPipe hand tracking and the custom gesture classification system.

Usage:
    python demos/hand_gesture_demo.py

Controls:
    - Press 'q' or 'ESC' to quit
    - Press 'i' to show detailed gesture information
    - Press 's' to save a screenshot

The system detects 28 Asamyuta Hastas (single hand gestures) from the
Bharathanatyam dataset and displays:
- Detected gesture name
- Confidence score
- Finger state analysis
- Gesture description and meanings
"""

import cv2
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hand import HandGestureDetector, BharathanatyamGestures


def draw_info_panel(img, detected_gestures, show_detailed_info=False):
    """
    Draw an information panel showing detected gestures and finger states.
    
    Args:
        img: Image to draw on.
        detected_gestures: List of detected gesture results.
        show_detailed_info: If True, show detailed finger state analysis.
    """
    panel_y = 10
    panel_x = 10
    line_height = 25
    
    # Draw semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x - 5, panel_y - 5), 
                  (panel_x + 400, panel_y + 200),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    y = panel_y + 20
    
    # Title
    cv2.putText(img, "BHARATHANATYAM HAND GESTURES", 
               (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += line_height + 10
    
    if not detected_gestures:
        cv2.putText(img, "No hands detected", 
                   (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return
    
    # Display each detected hand
    for idx, result in enumerate(detected_gestures):
        hand_label = result.get('hand_label', 'Unknown')
        gesture = result.get('gesture', 'Unknown')
        confidence = result.get('confidence', 0.0)
        
        # Hand label
        cv2.putText(img, f"Hand {idx + 1} ({hand_label}):", 
                   (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += line_height
        
        # Gesture name
        color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 0, 255)
        cv2.putText(img, f"Gesture: {gesture}", 
                   (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += line_height
        
        # Confidence
        cv2.putText(img, f"Confidence: {confidence:.1%}", 
                   (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += line_height
        
        # Detailed finger states
        if show_detailed_info and 'finger_states' in result:
            finger_states = result['finger_states']
            cv2.putText(img, "Finger States:", 
                       (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y += 20
            
            for finger, state in finger_states.items():
                if isinstance(state, bool):
                    state_text = "YES" if state else "NO"
                else:
                    state_text = state
                cv2.putText(img, f"  {finger}: {state_text}", 
                           (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
                y += 18
        
        y += 10  # Spacing between hands


def draw_controls_help(img):
    """Draw keyboard controls help at the bottom of the screen."""
    img_height, img_width, _ = img.shape
    
    help_text = "Controls: [q] Quit  [i] Info  [s] Screenshot"
    font_scale = 0.5
    thickness = 1
    
    # Calculate text size for background
    text_size = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (img_width - text_size[0]) // 2
    text_y = img_height - 15
    
    # Draw background
    cv2.rectangle(img, 
                  (text_x - 10, text_y - 20),
                  (text_x + text_size[0] + 10, text_y + 5),
                  (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(img, help_text, 
               (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
               font_scale, (255, 255, 255), thickness)


def main():
    """Main function to run the hand gesture detection demo."""
    print("=" * 60)
    print("BHARATHANATYAM HAND GESTURE DETECTION DEMO")
    print("=" * 60)
    
    # Initialize gesture detector
    print("\nInitializing hand gesture detector...")
    detector = HandGestureDetector(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Display loaded gesture information
    print(f"\n✓ Loaded {detector.gestures.get_total_gesture_count()} gestures:")
    print(f"  - {len(detector.gestures.asamyuta_hastas)} Asamyuta Hastas (single hand)")
    print(f"  - {len(detector.gestures.samyuta_hastas)} Samyuta Hastas (combined hands)")
    
    print("\nAsamyuta Hastas:")
    for i, name in enumerate(detector.gestures.get_all_asamyuta_names(), 1):
        print(f"  {i:2d}. {name}")
    
    print("\n" + "=" * 60)
    print("Starting webcam... Press 'q' to quit")
    print("=" * 60)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    show_info = False
    screenshot_count = 0
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect and classify hand gestures
        detected_gestures = detector.detect_gestures(frame, draw=True)
        
        # Draw information panel
        draw_info_panel(frame, detected_gestures, show_info)
        
        # Draw controls help
        draw_controls_help(frame)
        
        # Display the frame
        cv2.imshow('Bharathanatyam Hand Gesture Detection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key in [ord('q'), ord('Q'), 27]:  # 'q' or ESC
            break
        elif key in [ord('i'), ord('I')]:  # Toggle info
            show_info = not show_info
            print(f"Info display: {'ON' if show_info else 'OFF'}")
        elif key in [ord('s'), ord('S')]:  # Save screenshot
            screenshot_dir = project_root / "data" / "screenshots"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = screenshot_dir / f"gesture_{screenshot_count:03d}.png"
            cv2.imwrite(str(screenshot_path), frame)
            print(f"Screenshot saved: {screenshot_path}")
            screenshot_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo ended.")


if __name__ == "__main__":
    main()
