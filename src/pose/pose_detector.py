"""
PoseDetector Module
==================
This module provides a reusable class for detecting human body poses
using MediaPipe's Pose solution.

Each landmark has:
- x, y: Normalized coordinates (0.0 to 1.0) relative to image dimensions
- z: Depth from the hip center (lower z = closer to camera)
- visibility: Confidence score (0.0 to 1.0) that the landmark is visible
"""

import cv2  # OpenCV library for image processing
import mediapipe as mp  # Google's MediaPipe framework for ML pipelines
import numpy as np  # Numerical operations for angle calculations


class PoseDetector:
    """
    A class to detect and visualize human body poses from images/video frames.
    
    Attributes:
        model_complexity (int): 0=Lite, 1=Full (default), 2=Heavy model
        smooth_landmarks (bool): Whether to smooth landmark detections across frames
        detection_confidence (float): Minimum confidence for detection (0.0-1.0)
        tracking_confidence (float): Minimum confidence for tracking (0.0-1.0)
        mp_pose: MediaPipe Pose solution instance
        mp_drawing: MediaPipe drawing utilities
    """

    def __init__(
        self,
        model_complexity=1,
        smooth_landmarks=True,
        detection_confidence=0.5,
        tracking_confidence=0.5
    ):
        """
        Initialize the PoseDetector with MediaPipe Pose solution.
        
        Args:
            model_complexity (int): Model size/accuracy tradeoff.
                0 = Lite (fastest, least accurate)
                1 = Full (balanced) - DEFAULT
                2 = Heavy (slowest, most accurate)
            smooth_landmarks (bool): If True, filters landmark detections
                across frames to reduce jitter.
            detection_confidence (float): Minimum confidence (0.0-1.0)
                for person detection to be considered successful.
            tracking_confidence (float): Minimum confidence (0.0-1.0)
                for detected pose to be considered tracked in next frame.
        """
        # Store configuration parameters
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize MediaPipe Pose solution
        # mp.solutions.pose.Pose is the main pose detection model
        self.mp_pose = mp.solutions.pose.Pose(
            model_complexity=self.model_complexity,  # Controls model size
            smooth_landmarks=self.smooth_landmarks,  # Reduces jitter
            min_detection_confidence=self.detection_confidence,  # Detection threshold
            min_tracking_confidence=self.tracking_confidence  # Tracking threshold
        )

        # Initialize MediaPipe drawing utilities
        # mp.solutions.drawing_utils provides functions to draw landmarks & connections
        self.mp_drawing = mp.solutions.drawing_utils

        # Define drawing specifications for landmarks (dots on body joints)
        # color=green (0, 255, 0) in BGR format, thickness=2 pixels
        self.drawing_spec_landmarks = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green color for landmark dots (BGR format)
            thickness=2,  # Thickness of landmark dots
            circle_radius=2  # Radius of landmark circles
        )

        # Define drawing specifications for connections (lines between joints)
        # color=blue (255, 0, 0) in BGR format, thickness=2 pixels
        self.drawing_spec_connections = self.mp_drawing.DrawingSpec(
            color=(255, 0, 0),  # Blue color for connection lines (BGR format)
            thickness=2,  # Thickness of connection lines
            circle_radius=1  # Radius at connection endpoints
        )

    def find_pose(self, img, draw=True):
        """
        Detect pose landmarks in an image/frame.
        
        This method:
        1. Converts the image from BGR (OpenCV format) to RGB (MediaPipe format)
        2. Runs the pose detection model
        3. Stores results in self.results for later use
        4. Optionally draws landmarks and connections on the image
        
        Args:
            img (numpy.ndarray): Input image/frame in BGR format from OpenCV.
                Shape: (height, width, 3) where channels are B, G, R.
            draw (bool): If True, draws landmarks and connections on the image.
                Default is True for visualization.
        
        Returns:
            numpy.ndarray: The image with (optionally) drawn pose landmarks.
                Same shape as input, modified in-place if draw=True.
        """
        # Convert image from BGR (OpenCV) to RGB (MediaPipe expects RGB)
        # OpenCV loads images in BGR channel order, but ML models expect RGB
        # cv2.cvtColor() converts color spaces
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Pose model
        # self.mp_pose.process() runs the ML model on the image
        # Returns a Results object containing detected landmarks
        self.results = self.mp_pose.process(img_rgb)

        # If draw=True and landmarks were detected, visualize them
        # self.results.pose_landmarks is None if no pose was detected
        if self.results.pose_landmarks is not None and draw:
            # Draw pose landmarks (dots on body joints) and connections (lines)
            # self.mp_drawing.draw_landmarks() draws on the original image
            # It uses POSE_CONNECTIONS to know which landmarks to connect
            self.mp_drawing.draw_landmarks(
                img,  # Image to draw on (modified in-place)
                self.results.pose_landmarks,  # Detected landmark positions
                # POSE_CONNECTIONS defines which landmarks are connected
                mp.solutions.pose.POSE_CONNECTIONS,
                # Style for landmark dots (overridden by landmark_drawing_spec)
                landmark_drawing_spec=self.drawing_spec_landmarks,
                # Style for connection lines
                connection_drawing_spec=self.drawing_spec_connections
            )

        # Return the image (with or without drawings)
        return img

    def get_landmarks(self):
        """
        Get the detected pose landmarks as a list of coordinates.
        
        This method extracts landmark data from the most recent detection.
        Each landmark contains: x, y, z, visibility
        
        Returns:
            list[dict] or None: List of landmark dictionaries, one per landmark.
                Each dict has keys: 'x', 'y', 'z', 'visibility'
                Returns None if no pose was detected in the last frame.
        
        Example:
            landmarks = detector.get_landmarks()
            if landmarks:
                nose = landmarks[0]  # Nose is landmark index 0
                print(f"Nose position: x={nose['x']:.3f}, y={nose['y']:.3f}")
        """
        # Check if pose was detected in the last processed frame
        # results.pose_landmarks is None if no person was detected
        if self.results.pose_landmarks is None:
            return None  # No pose detected, return None

        # Extract landmark data into a clean, usable Python list
        # self.results.pose_landmarks.landmark contains all 33 landmarks
        landmarks = []
        for landmark in self.results.pose_landmarks.landmark:
            # Create a dictionary for each landmark with its properties
            landmarks.append({
                'x': landmark.x,  # Normalized x position (0.0 to 1.0)
                'y': landmark.y,  # Normalized y position (0.0 to 1.0)
                'z': landmark.z,  # Depth relative to hip center
                'visibility': landmark.visibility  # Visibility confidence (0-1)
            })

        # Return the list of landmark dictionaries
        return landmarks

    @staticmethod
    def calculate_angle(p1, p2, p3):
        """
        Calculate the angle between three landmarks (p1-p2-p3).
        
        The angle is calculated at point p2 (the middle point).
        Uses the dot product formula: cos(theta) = (a·b) / (|a|*|b|)
        
        Args:
            p1 (dict): First landmark with 'x', 'y' keys.
            p2 (dict): Middle landmark (vertex) with 'x', 'y' keys.
            p3 (dict): Third landmark with 'x', 'y' keys.
            
        Returns:
            float: Angle in degrees (0-180).
            
        Example:
            # Calculate elbow angle (shoulder-elbow-wrist)
            angle = detector.calculate_angle(shoulder, elbow, wrist)
        """
        # Convert landmark coordinates to numpy arrays
        # We use x and y coordinates (can also include z for 3D angles)
        a = np.array([p1['x'], p1['y']])  # First point
        b = np.array([p2['x'], p2['y']])  # Middle point (vertex)
        c = np.array([p3['x'], p3['y']])  # Third point
        
        # Calculate vectors from the vertex (point b)
        ba = a - b  # Vector from b to a
        bc = c - b  # Vector from b to c
        
        # Calculate the dot product of the two vectors
        # dot_product = ba_x * bc_x + ba_y * bc_y
        dot_product = np.dot(ba, bc)
        
        # Calculate the magnitude (length) of each vector
        # magnitude = sqrt(x^2 + y^2)
        magnitude_ba = np.linalg.norm(ba)
        magnitude_bc = np.linalg.norm(bc)
        
        # Avoid division by zero if points are too close
        if magnitude_ba == 0 or magnitude_bc == 0:
            return 0.0
        
        # Calculate cosine of the angle using dot product formula
        # cos(theta) = (a·b) / (|a|*|b|)
        # np.clip ensures the value stays in valid range [-1, 1]
        cos_theta = np.clip(dot_product / (magnitude_ba * magnitude_bc), -1.0, 1.0)
        
        # Calculate angle in radians and convert to degrees
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

    def draw_joint_angles(self, img, angles_config=None):
        """
        Calculate and draw joint angles on the image.
        
        Each angle is displayed at the vertex point with a colored label.
        
        Args:
            img (numpy.ndarray): Image to draw on (BGR format).
            angles_config (list[dict], optional): List of angle configurations.
                Each dict has keys:
                - 'p1': landmark index for first point
                - 'p2': landmark index for vertex (middle point)
                - 'p3': landmark index for third point
                - 'label': name of the angle to display
                
                If None, uses default common joint angles.
                
        Returns:
            numpy.ndarray: Image with drawn joint angles.
            
        Example:
            config = [
                {'p1': LEFT_SHOULDER, 'p2': LEFT_ELBOW, 'p3': LEFT_WRIST, 'label': 'L Elbow'},
                {'p1': RIGHT_SHOULDER, 'p2': RIGHT_ELBOW, 'p3': RIGHT_WRIST, 'label': 'R Elbow'}
            ]
            img = detector.draw_joint_angles(img, config)
        """
        # Import landmark constants
        from landmarks import (
            LEFT_SHOULDER, RIGHT_SHOULDER,
            LEFT_ELBOW, RIGHT_ELBOW,
            LEFT_WRIST, RIGHT_WRIST,
            LEFT_HIP, RIGHT_HIP,
            LEFT_KNEE, RIGHT_KNEE,
            LEFT_ANKLE, RIGHT_ANKLE
        )
        
        # Default configuration for common joint angles
        if angles_config is None:
            angles_config = [
                # Upper body angles
                {'p1': LEFT_SHOULDER, 'p2': LEFT_ELBOW, 'p3': LEFT_WRIST, 'label': 'L Elbow'},
                {'p1': RIGHT_SHOULDER, 'p2': RIGHT_ELBOW, 'p3': RIGHT_WRIST, 'label': 'R Elbow'},
                {'p1': LEFT_ELBOW, 'p2': LEFT_SHOULDER, 'p3': LEFT_HIP, 'label': 'L Shoulder'},
                {'p1': RIGHT_ELBOW, 'p2': RIGHT_SHOULDER, 'p3': RIGHT_HIP, 'label': 'R Shoulder'},
                # Lower body angles
                {'p1': LEFT_HIP, 'p2': LEFT_KNEE, 'p3': LEFT_ANKLE, 'label': 'L Knee'},
                {'p1': RIGHT_HIP, 'p2': RIGHT_KNEE, 'p3': RIGHT_ANKLE, 'label': 'R Knee'},
                {'p1': LEFT_SHOULDER, 'p2': LEFT_HIP, 'p3': LEFT_KNEE, 'label': 'L Hip'},
                {'p1': RIGHT_SHOULDER, 'p2': RIGHT_HIP, 'p3': RIGHT_KNEE, 'label': 'R Hip'},
            ]
        
        # Get image dimensions to scale normalized coordinates
        img_height, img_width, _ = img.shape
        
        # Get landmarks
        landmarks = self.get_landmarks()
        if landmarks is None:
            return img  # No pose detected
        
        # Calculate and draw each angle
        for angle_def in angles_config:
            p1_idx = angle_def['p1']
            p2_idx = angle_def['p2']
            p3_idx = angle_def['p3']
            label = angle_def['label']
            
            # Get the three landmarks
            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            p3 = landmarks[p3_idx]
            
            # Check visibility (only draw if vertex is visible enough)
            if p2['visibility'] < 0.5:
                continue
            
            # Calculate angle
            angle_value = self.calculate_angle(p1, p2, p3)
            
            # Convert vertex coordinates to pixel position
            cx = int(p2['x'] * img_width)
            cy = int(p2['y'] * img_height)
            
            # Determine color based on angle (green for good, red for extreme)
            # Normal range: 30-150 degrees, outside this = warning color
            if 30 <= angle_value <= 150:
                color = (0, 255, 0)  # Green (BGR)
            else:
                color = (0, 0, 255)  # Red (BGR)
            
            # Draw angle label on image
            # Position slightly above the vertex
            text_position = (cx + 10, cy - 10)
            
            # Format angle text with 1 decimal place
            text = f"{label}: {angle_value:.1f}°"
            
            # Draw a semi-transparent background rectangle for better readability
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                img,
                (text_position[0] - 2, text_position[1] - 15),
                (text_position[0] + text_size[0] + 2, text_position[1] + text_size[1] + 2),
                (0, 0, 0),  # Black background
                -1  # Filled rectangle
            )
            
            # Draw the text
            cv2.putText(
                img,
                text,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        return img

    def draw_specific_landmarks(self, img, landmark_indices):
        """
        Draw specific landmark points with labels on the image.
        
        Useful for highlighting specific body parts during analysis.
        
        Args:
            img (numpy.ndarray): Image to draw on (BGR format).
            landmark_indices (list[int]): Indices of landmarks to draw.
                See POSE_LANDMARKS for index mapping.
        
        Returns:
            numpy.ndarray: Image with drawn landmarks.
        """
        # Get image dimensions to scale normalized coordinates
        img_height, img_width, _ = img.shape

        # Check if pose was detected
        if self.results.pose_landmarks is None:
            return img  # Return unchanged image if no pose

        # Draw each requested landmark
        for idx in landmark_indices:
            # Get the landmark at this index
            landmark = self.results.pose_landmarks.landmark[idx]

            # Convert normalized coordinates (0-1) to pixel coordinates
            # landmark.x and landmark.y are normalized (0.0 to 1.0)
            # Multiply by image width/height to get actual pixel positions
            cx = int(landmark.x * img_width)  # Center x in pixels
            cy = int(landmark.y * img_height)  # Center y in pixels

            # Draw a red circle at the landmark position
            # cv2.circle() draws a filled circle if thickness=-1
            cv2.circle(
                img,  # Image to draw on
                (cx, cy),  # Center coordinates (x, y)
                5,  # Radius in pixels
                (0, 0, 255),  # Color: Red in BGR format
                -1  # Thickness: -1 means filled circle
            )

            # Add text label showing the landmark index
            # cv2.putText() draws text on the image
            cv2.putText(
                img,  # Image to draw on
                str(idx),  # Text to display (landmark index as string)
                (cx + 10, cy - 10),  # Position: 10px right, 10px above center
                cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                0.5,  # Font scale (size)
                (255, 0, 0),  # Color: Blue in BGR format
                1  # Thickness: 1 pixel
            )

        # Return the modified image
        return img
