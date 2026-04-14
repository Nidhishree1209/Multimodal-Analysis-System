"""
Run pose detection using webcam with joint angle display.
"""
import cv2
from pose_detector import PoseDetector


def main():
    # Initialize pose detector
    detector = PoseDetector()

    # Open webcam (0 = default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    print("Joint angles will be displayed on screen.")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect pose and draw landmarks
        frame = detector.find_pose(frame, draw=True)
        
        # Draw joint angles on the frame
        frame = detector.draw_joint_angles(frame)

        # Display the result
        cv2.imshow('Pose Detection with Joint Angles', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
