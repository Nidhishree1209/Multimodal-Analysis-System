"""
Pose Detection Demo
===================
This script demonstrates real-time pose detection using your webcam.

WHAT THIS DOES:
1. Opens your webcam using OpenCV
2. Detects body poses using MediaPipe Pose
3. Draws skeleton (joints + connections) on your body in real-time
4. Displays landmark positions in the console
5. Press 'q' to quit, 'l' to toggle landmark labels

HOW TO RUN:
    python demos/pose_detection_demo.py

KEY CONCEPTS DEMONSTRATED:
- Real-time video processing with OpenCV
- MediaPipe Pose detection pipeline
- Landmark visualization
- Frame-by-frame processing loop
"""

import cv2  # OpenCV library for camera access and image processing
import sys  # System module for path manipulation
import time  # Time module for FPS calculation

# Add the project root to Python path so we can import our modules
# sys.path.insert() adds a directory to the front of Python's module search path
# This allows us to import 'src' modules even though we're in the 'demos' folder
sys.path.insert(0, '.')  # Add current directory (project root) to path

# Import our custom PoseDetector class from the src/pose module
# The dot (.) in 'from src.pose' means relative import from current directory
from src.pose.pose_detector import PoseDetector
from src.pose.landmarks import NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST


def main():
    """
    Main function that runs the pose detection demo.
    
    This function:
    1. Initializes the webcam
    2. Creates a PoseDetector instance
    3. Runs a loop that processes each frame
    4. Displays the result in a window
    5. Handles keyboard input to control the demo
    """
    # --- STEP 1: Initialize the webcam ---
    # cv2.VideoCapture(0) opens the default camera (index 0 = first camera)
    # You can change 0 to 1, 2, etc. if you have multiple cameras
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    # cap.isOpened() returns True if camera is accessible, False otherwise
    if not cap.isOpened():
        # If camera can't be opened, print error and exit
        print("ERROR: Could not open camera")
        print("Make sure your webcam is connected and working")
        return  # Exit the function (and the program)

    # --- STEP 2: Create PoseDetector instance ---
    # model_complexity=1 uses the full model (balanced speed/accuracy)
    # detection_confidence=0.5 means we need 50% confidence to detect a person
    detector = PoseDetector(
        model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
        smooth_landmarks=True,  # Smooth detections across frames
        detection_confidence=0.5,  # Minimum detection confidence
        tracking_confidence=0.5  # Minimum tracking confidence
    )

    # --- STEP 3: Configuration variables ---
    show_labels = False  # Toggle for showing landmark index labels
    frame_count = 0  # Counter for FPS calculation
    start_time = time.time()  # Record start time for FPS calculation
    fps = 0  # Frames per second (updated periodically)

    # --- STEP 4: Main processing loop ---
    # while True creates an infinite loop (we'll break out with 'q' key)
    # Each iteration processes one frame from the camera
    while True:
        # Read a frame from the camera
        # cap.read() returns two values:
        #   success (bool): True if frame was read successfully
        #   frame (numpy.ndarray): The actual image data (BGR format)
        success, frame = cap.read()

        # Check if frame was read successfully
        if not success:
            # If frame reading failed, print warning and skip this frame
            print("WARNING: Could not read frame")
            continue  # Skip to next iteration of the loop

        # --- STEP 5: Detect pose in the frame ---
        # detector.find_pose() processes the frame and returns it with drawings
        # draw=True (default) means it will draw the skeleton on the frame
        frame = detector.find_pose(frame, draw=True)

        # Optionally draw specific landmark labels (toggle with 'l' key)
        if show_labels:
            # Draw labels for key landmarks: nose, shoulders, wrists
            frame = detector.draw_specific_landmarks(
                frame,
                [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST]
            )

        # --- STEP 6: Get and display landmark data ---
        # detector.get_landmarks() returns list of landmark dictionaries
        landmarks = detector.get_landmarks()

        # Check if pose was detected (landmarks is None if no person detected)
        if landmarks is not None:
            # Example: Print nose position every 30 frames
            # frame_count % 30 == 0 means "every 30th frame"
            if frame_count % 30 == 0:
                # Get nose landmark (index 0)
                nose = landmarks[NOSE]
                # Print formatted position info
                # f-string allows embedding expressions inside string literals
                # :.3f means "format as float with 3 decimal places"
                print(f"Nose position: x={nose['x']:.3f}, y={nose['y']:.3f}, "
                      f"z={nose['z']:.3f}, visibility={nose['visibility']:.2f}")

        # --- STEP 7: Calculate and display FPS ---
        # Increment frame counter
        frame_count += 1

        # Recalculate FPS every 30 frames
        if frame_count % 30 == 0:
            # Calculate elapsed time since start
            elapsed_time = time.time() - start_time
            # FPS = total frames / elapsed time
            fps = frame_count / elapsed_time
            # Reset counters for next calculation period
            frame_count = 0
            start_time = time.time()

        # Draw FPS text on the frame
        # cv2.putText() draws text on an image
        cv2.putText(
            frame,  # Image to draw on
            f"FPS: {fps:.1f}",  # Text to display (formatted to 1 decimal)
            (10, 30),  # Position: top-left corner (10px from left, 30px from top)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type (simple sans-serif)
            1.0,  # Font scale (1.0 = normal size)
            (0, 255, 0),  # Color: Green in BGR format
            2  # Thickness: 2 pixels
        )

        # --- STEP 8: Display the frame ---
        # cv2.imshow() creates a window and displays the image
        # "Pose Detection Demo" is the window title
        cv2.imshow("Pose Detection Demo", frame)

        # --- STEP 9: Handle keyboard input ---
        # cv2.waitKey(1) waits 1 millisecond for a key press
        # Returns the ASCII code of the pressed key, or -1 if no key pressed
        # We use & 0xFF to handle different keyboard layouts correctly
        key = cv2.waitKey(1) & 0xFF

        # Check which key was pressed
        if key == ord('q'):
            # ord('q') returns ASCII code for 'q' key
            # If user pressed 'q', break out of the loop (quit)
            print("Quitting demo...")
            break  # Exit the while loop
        elif key == ord('l'):
            # If user pressed 'l', toggle landmark label display
            show_labels = not show_labels  # Flip True <-> False
            # Print current state
            state = "ON" if show_labels else "OFF"
            print(f"Landmark labels: {state}")

    # --- STEP 10: Cleanup ---
    # This code runs after the loop breaks (when user presses 'q')

    # Release the camera (free up the device for other programs)
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Print completion message
    print("Demo ended successfully!")


# This conditional checks if the script is being run directly
# (as opposed to being imported as a module)
# __name__ is a special Python variable that equals "__main__" when run directly
if __name__ == "__main__":
    # Call the main function to start the demo
    main()
