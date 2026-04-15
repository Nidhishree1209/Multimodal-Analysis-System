"""
HandGestureDetector Module
==========================
This module provides a comprehensive class for detecting and classifying
Bharathanatyam hand gestures (hastas) using MediaPipe's Hand solution.

It analyzes finger states (straight, bent, folded, touching) and matches
them against detection rules from the Bharathanatyam dataset.

MediaPipe Hand Landmarks (21 points per hand):
0: Wrist
1-4: Thumb (CMC, MCP, IP, TIP)
5-8: Index finger (MCP, PIP, DIP, TIP)
9-12: Middle finger (MCP, PIP, DIP, TIP)
13-16: Ring finger (MCP, PIP, DIP, TIP)
17-20: Pinky finger (MCP, PIP, DIP, TIP)
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple

from .gesture_definitions import BharathanatyamGestures
from .gesture_definitions import (
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
    INDEX_MCP, INDEX_PIP, INDEX_DIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP,
    RING_MCP, RING_PIP, RING_DIP,
    PINKY_MCP, PINKY_PIP, PINKY_DIP,
    THUMB_MCP, THUMB_IP
)


class HandGestureDetector:
    """
    A class to detect and classify Bharathanatyam hand gestures.
    
    Attributes:
        mp_hands (mp.solutions.hands.Hands): MediaPipe Hands solution instance
        mp_drawing (mp.solutions.drawing_utils): MediaPipe drawing utilities
        gestures (BharathanatyamGestures): Gesture definitions loader
        max_num_hands (int): Maximum number of hands to detect (default: 2)
        confidence thresholds for detection and tracking
    """
    
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        dataset_dir: Optional[str] = None
    ):
        """
        Initialize HandGestureDetector.

        Args:
            max_num_hands (int): Maximum number of hands to track (1 or 2).
            min_detection_confidence (float): Minimum confidence for hand detection.
            min_tracking_confidence (float): Minimum confidence for hand tracking.
            dataset_dir (str, optional): Path to dataset directory.
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe Hands solution
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize the Hands solution ONCE here
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

        # Custom drawing specifications for hand landmarks
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green for landmarks
            thickness=2,
            circle_radius=3
        )

        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(255, 0, 0),  # Blue for connections
            thickness=2,
            circle_radius=1
        )

        # Load Bharathanatyam gesture definitions
        self.gestures = BharathanatyamGestures(dataset_dir)

        # Store latest hand landmarks
        self.hand_landmarks = []  # List of landmarks for each detected hand
        self.hand_labels = []  # 'Left' or 'Right' for each hand
        self.detected_gestures = []  # Detected gesture names for each hand
        
        # NEW: Temporal smoothing for stable gesture detection
        self._gesture_history = {}  # Track recent gestures per hand
        self._smoothing_window = 5  # Number of frames to smooth over
        self._min_stable_confidence = 0.6  # Minimum confidence to keep gesture
    
    def find_hands(self, img, draw=True):
        """
        Detect hand landmarks in an image/frame.

        Args:
            img (numpy.ndarray): Input image/frame in BGR format.
            draw (bool): If True, draws landmarks and connections on the image.

        Returns:
            numpy.ndarray: The image with (optionally) drawn hand landmarks.
        """
        # Convert BGR to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Hands (reuse the existing instance)
        results = self.hands.process(img_rgb)
        
        # Store hand landmarks for later analysis
        self.hand_landmarks = []
        self.hand_labels = []
        
        # Draw and collect landmarks
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Store landmarks
                self.hand_landmarks.append(hand_landmarks.landmark)
                
                # Get hand label (Left/Right)
                if results.multi_handedness:
                    hand_label = results.multi_handedness[idx].classification[0].label
                    self.hand_labels.append(hand_label)
                
                # Draw landmarks if requested
                if draw:
                    self.mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.landmark_drawing_spec,
                        self.connection_drawing_spec
                    )
        
        return img
    
    @staticmethod
    def _calculate_distance(landmark1, landmark2) -> float:
        """
        Calculate Euclidean distance between two landmarks.
        
        Args:
            landmark1, landmark2: Landmark objects with x, y, z coordinates.
        
        Returns:
            float: Euclidean distance between the two landmarks.
        """
        return np.sqrt(
            (landmark1.x - landmark2.x) ** 2 +
            (landmark1.y - landmark2.y) ** 2 +
            (landmark1.z - landmark2.z) ** 2
        )
    
    @staticmethod
    def _calculate_angle(p1, p2, p3) -> float:
        """
        Calculate angle between three points (p1-p2-p3).
        
        Args:
            p1, p2, p3: Points with x, y, z coordinates.
            
        Returns:
            float: Angle in degrees (0-180).
        """
        import math
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        return angle

    @staticmethod
    def _is_finger_extended(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx) -> bool:
        """
        Check if a finger is fully extended (straight).

        A finger is extended if all joints are nearly aligned (angles > 150 degrees).
        IMPROVED: More lenient threshold to handle detection noise.

        Args:
            landmarks: List of hand landmarks.
            mcp_idx, pip_idx, dip_idx, tip_idx: Indices for finger joints.

        Returns:
            bool: True if finger is extended.
        """
        angle_pip = HandGestureDetector._calculate_angle(
            landmarks[mcp_idx], landmarks[pip_idx], landmarks[dip_idx]
        )
        angle_dip = HandGestureDetector._calculate_angle(
            landmarks[pip_idx], landmarks[dip_idx], landmarks[tip_idx]
        )

        # IMPROVED: Finger is extended if both joints are nearly straight (>150 degrees)
        # Lowered from 160 to 150 for better detection
        return angle_pip > 150 and angle_dip > 150

    @staticmethod
    def _is_finger_folded(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx) -> bool:
        """
        Check if a finger is folded (curled into palm).

        A finger is folded if the tip is close to or below the PIP joint level,
        indicating the finger is curled. IMPROVED: Better threshold for folding.

        Args:
            landmarks: List of hand landmarks.
            mcp_idx, pip_idx, dip_idx, tip_idx: Indices for finger joints.

        Returns:
            bool: True if finger is folded.
        """
        angle_pip = HandGestureDetector._calculate_angle(
            landmarks[mcp_idx], landmarks[pip_idx], landmarks[dip_idx]
        )
        angle_dip = HandGestureDetector._calculate_angle(
            landmarks[pip_idx], landmarks[dip_idx], landmarks[tip_idx]
        )

        # IMPROVED: Finger is folded if average joint angle is significantly bent (< 100°)
        # Tightened from 110 to 100 for better distinction from 'bent'
        avg_angle = (angle_pip + angle_dip) / 2.0

        # Also check if tip is below MCP (y-coordinate)
        mcp = landmarks[mcp_idx]
        tip = landmarks[tip_idx]
        tip_below_mcp = tip.y > mcp.y

        return avg_angle < 100 or (avg_angle < 120 and tip_below_mcp)

    @staticmethod
    def _is_finger_curved(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx) -> bool:
        """
        Check if a finger is curved (as if grasping a ball).
        
        Curved means moderate bend at both joints (90-150 degrees).

        Args:
            landmarks: List of hand landmarks.
            mcp_idx, pip_idx, dip_idx, tip_idx: Indices for finger joints.

        Returns:
            bool: True if finger is curved.
        """
        angle_pip = HandGestureDetector._calculate_angle(
            landmarks[mcp_idx], landmarks[pip_idx], landmarks[dip_idx]
        )
        angle_dip = HandGestureDetector._calculate_angle(
            landmarks[pip_idx], landmarks[dip_idx], landmarks[tip_idx]
        )
        
        # Curved: moderate bend at both joints
        return 90 <= angle_pip <= 160 and 90 <= angle_dip <= 160

    @staticmethod
    def _is_finger_slightly_bent(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx) -> bool:
        """
        Check if a finger is slightly bent (minor bend, 150-170 degrees).

        Args:
            landmarks: List of hand landmarks.
            mcp_idx, pip_idx, dip_idx, tip_idx: Indices for finger joints.

        Returns:
            bool: True if finger is slightly bent.
        """
        angle_pip = HandGestureDetector._calculate_angle(
            landmarks[mcp_idx], landmarks[pip_idx], landmarks[dip_idx]
        )
        angle_dip = HandGestureDetector._calculate_angle(
            landmarks[pip_idx], landmarks[dip_idx], landmarks[tip_idx]
        )
        
        # Slightly bent: nearly straight but not fully
        return 150 <= angle_pip < 160 or 150 <= angle_dip < 160

    def _is_finger_bent(self, landmarks, mcp_idx, pip_idx, dip_idx, tip_idx) -> bool:
        """
        Check if a finger is bent (partially folded, distinct from folded/curved).

        Args:
            landmarks: List of hand landmarks.
            mcp_idx, pip_idx, dip_idx, tip_idx: Indices for finger joints.

        Returns:
            bool: True if finger is bent.
        """
        angle_pip = HandGestureDetector._calculate_angle(
            landmarks[mcp_idx], landmarks[pip_idx], landmarks[dip_idx]
        )
        angle_dip = HandGestureDetector._calculate_angle(
            landmarks[pip_idx], landmarks[dip_idx], landmarks[tip_idx]
        )
        
        # Bent: moderate to significant bend, but not folded
        # Average angle between 110° and 150°
        avg_angle = (angle_pip + angle_dip) / 2.0
        
        return 110 <= avg_angle < 150 and not self._is_finger_folded(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx)
    
    def _calculate_hand_size(self, landmarks) -> float:
        """
        Calculate the size of the hand in normalized coordinates.
        Used to adapt distance thresholds for different hand sizes/distances from camera.
        
        Args:
            landmarks: Hand landmarks.
            
        Returns:
            float: Hand size (distance from wrist to middle finger MCP).
        """
        return self._calculate_distance(landmarks[WRIST], landmarks[MIDDLE_MCP])
    
    def _are_fingers_touching(self, landmarks, finger1_tip_idx, finger2_tip_idx, threshold=None) -> bool:
        """
        Check if two finger tips are touching.
        
        Uses adaptive threshold based on hand size for better accuracy.

        Args:
            landmarks: List of hand landmarks.
            finger1_tip_idx, finger2_tip_idx: Indices of finger tips.
            threshold: Distance threshold (if None, uses adaptive threshold).

        Returns:
            bool: True if fingers are touching.
        """
        distance = self._calculate_distance(landmarks[finger1_tip_idx], landmarks[finger2_tip_idx])
        
        # Use adaptive threshold based on hand size
        if threshold is None:
            hand_size = self._calculate_hand_size(landmarks)
            # Threshold is 12% of hand size (adaptive to hand distance from camera)
            threshold = hand_size * 0.12
        
        return distance < threshold
    
    def analyze_finger_states(self, landmarks) -> Dict[str, str]:
        """
        Analyze the state of each finger in the hand.

        Finger states detection hierarchy:
        1. Check if fingers are touching thumb first (overrides other states)
        2. Check for 'straight' (fully extended, angles > 160°)
        3. Check for 'slightly_bent' (nearly straight, 150-170°)
        4. Check for 'curved' (moderate bend, 90-150°)
        5. Check for 'bent' (significant bend, <150° but not folded)
        6. Check for 'folded' (curled into palm, angles < 90°)

        Args:
            landmarks: List of hand landmarks for one hand.

        Returns:
            dict: Dictionary mapping finger names to their states.
                Keys: 'thumb', 'index', 'middle', 'ring', 'pinky'
        """
        states = {}

        # ===== Check thumb-to-finger touching relationships first =====
        thumb_touching_index = self._are_fingers_touching(landmarks, THUMB_TIP, INDEX_TIP)
        thumb_touching_middle = self._are_fingers_touching(landmarks, THUMB_TIP, MIDDLE_TIP)
        thumb_touching_ring = self._are_fingers_touching(landmarks, THUMB_TIP, RING_TIP)
        thumb_touching_pinky = self._are_fingers_touching(landmarks, THUMB_TIP, PINKY_TIP)
        
        # Check if thumb is touching base of ring finger (Chatura Hasta)
        thumb_to_ring_base_dist = self._calculate_distance(landmarks[THUMB_TIP], landmarks[RING_MCP])
        thumb_touching_ring_base = thumb_to_ring_base_dist < 0.08
        
        # Count how many fingers are touching thumb
        touching_count = sum([thumb_touching_index, thumb_touching_middle, thumb_touching_ring, thumb_touching_pinky])

        # ===== Analyze index finger =====
        if not thumb_touching_index:  # Only check extension if not touching thumb
            if self._is_finger_extended(landmarks, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP):
                states['index'] = 'straight'
            elif self._is_finger_slightly_bent(landmarks, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP):
                states['index'] = 'slightly_bent'
            elif self._is_finger_curved(landmarks, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP):
                states['index'] = 'curved'
            elif self._is_finger_bent(landmarks, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP):
                states['index'] = 'bent'
            else:
                states['index'] = 'folded'
        else:
            states['index'] = 'touching_thumb'

        # ===== Analyze middle finger =====
        if not thumb_touching_middle:
            if self._is_finger_extended(landmarks, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP):
                states['middle'] = 'straight'
            elif self._is_finger_slightly_bent(landmarks, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP):
                states['middle'] = 'slightly_bent'
            elif self._is_finger_curved(landmarks, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP):
                states['middle'] = 'curved'
            elif self._is_finger_bent(landmarks, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP):
                states['middle'] = 'bent'
            else:
                states['middle'] = 'folded'
        else:
            states['middle'] = 'touching_thumb'

        # ===== Analyze ring finger =====
        if not thumb_touching_ring:
            if self._is_finger_extended(landmarks, RING_MCP, RING_PIP, RING_DIP, RING_TIP):
                states['ring'] = 'straight'
            elif self._is_finger_slightly_bent(landmarks, RING_MCP, RING_PIP, RING_DIP, RING_TIP):
                states['ring'] = 'slightly_bent'
            elif self._is_finger_curved(landmarks, RING_MCP, RING_PIP, RING_DIP, RING_TIP):
                states['ring'] = 'curved'
            elif self._is_finger_bent(landmarks, RING_MCP, RING_PIP, RING_DIP, RING_TIP):
                states['ring'] = 'bent'
            else:
                states['ring'] = 'folded'
        else:
            states['ring'] = 'touching_thumb'

        # ===== Analyze pinky finger =====
        if not thumb_touching_pinky:
            if self._is_finger_extended(landmarks, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP):
                states['pinky'] = 'straight'
            elif self._is_finger_slightly_bent(landmarks, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP):
                states['pinky'] = 'slightly_bent'
            elif self._is_finger_curved(landmarks, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP):
                states['pinky'] = 'curved'
            elif self._is_finger_bent(landmarks, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP):
                states['pinky'] = 'bent'
            else:
                states['pinky'] = 'folded'
        else:
            states['pinky'] = 'touching_thumb'

        # ===== Analyze thumb state =====
        thumb_tip = landmarks[THUMB_TIP]
        thumb_mcp = landmarks[THUMB_MCP]
        thumb_ip = landmarks[THUMB_IP]
        wrist = landmarks[WRIST]
        
        # Calculate thumb extension angle
        thumb_angle = self._calculate_angle(wrist, thumb_mcp, thumb_tip)
        
        # Thumb extension based on distance from wrist
        thumb_extension = self._calculate_distance(thumb_tip, wrist)
        index_mcp = landmarks[INDEX_MCP]
        thumb_spread = self._calculate_distance(thumb_tip, index_mcp)
        
        if touching_count >= 3:
            # Thumb touching multiple fingertips
            if touching_count == 4:
                states['thumb'] = 'touching_all_fingertips'
            elif thumb_touching_middle and thumb_touching_ring and thumb_touching_pinky:
                states['thumb'] = 'touching_middle_ring_pinky'
            elif thumb_touching_index and thumb_touching_middle:
                states['thumb'] = 'touching_index_and_middle'
            elif thumb_touching_middle and thumb_touching_ring:
                states['thumb'] = 'touching_middle_and_ring'
            elif thumb_touching_ring and thumb_touching_pinky:
                states['thumb'] = 'touching_ring_and_pinky'
        elif thumb_touching_ring_base:
            states['thumb'] = 'touching_base_of_ring'
        elif thumb_touching_index:
            states['thumb'] = 'touching_index'
        elif thumb_touching_middle:
            states['thumb'] = 'touching_middle'
        elif thumb_touching_ring:
            states['thumb'] = 'touching_ring'
        elif thumb_touching_pinky:
            states['thumb'] = 'touching_pinky'
        elif thumb_extension > 0.25 and thumb_angle > 100:
            # Thumb fully extended outward
            states['thumb'] = 'stretched_out'
        elif thumb_spread > 0.15 and thumb_angle > 90:
            # Thumb spread wide
            states['thumb'] = 'straight_spread'
        elif thumb_angle < 60:
            # Thumb folded over fingers
            states['thumb'] = 'folded_over_fingers'
        elif 60 <= thumb_angle <= 100:
            # Thumb slightly bent or curved
            if self._is_finger_curved(landmarks, THUMB_MCP, THUMB_IP, THUMB_TIP, THUMB_TIP):
                states['thumb'] = 'curved'
            else:
                states['thumb'] = 'straight'
        else:
            states['thumb'] = 'straight'

        # ===== Check for special finger relationships =====
        
        # Calculate hand size for adaptive thresholds
        hand_size = self._calculate_hand_size(landmarks)
        
        # Check finger spread (V-shape detection for Kartareemukha Hasta)
        index_middle_spread = self._calculate_distance(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP])
        # V-shape threshold: 40% of hand size
        if index_middle_spread > hand_size * 0.40:
            states['index_middle_spread'] = True
        
        # Check if pinky is spread/angled out (Chatura Hasta)
        ring_pinky_spread = self._calculate_distance(landmarks[RING_TIP], landmarks[PINKY_TIP])
        # Pinky spread threshold: 35% of hand size
        if ring_pinky_spread > hand_size * 0.35 and states.get('pinky') == 'straight':
            states['pinky'] = 'straight_angled_out'
        
        # Check if index/middle are bent forward (Mrugasheersha, Tamrachooda)
        # This is detected when fingers are bent but tips are forward of MCP
        if states.get('index') in ['bent', 'slightly_bent']:
            if landmarks[INDEX_TIP].y < landmarks[INDEX_MCP].y:
                # Tip is above MCP (forward in camera space)
                if index_middle_spread > hand_size * 0.20:
                    states['index'] = 'bent_forward_separated'
                else:
                    states['index'] = 'bent_forward'
        
        if states.get('middle') in ['bent', 'slightly_bent']:
            if landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_MCP].y:
                states['middle'] = 'bent_forward'

        return states
    
    def _is_right_hand(self, landmarks) -> bool:
        """
        Determine if landmarks belong to a right hand based on thumb position.
        
        Args:
            landmarks: Hand landmarks.
        
        Returns:
            bool: True if right hand.
        """
        # Compare thumb tip x-coordinate with wrist x-coordinate
        # For right hand, thumb should be to the left of wrist (lower x)
        return landmarks[THUMB_TIP].x < landmarks[WRIST].x
    
    def classify_gesture(self, landmarks, hand_label: str = 'Right') -> Dict:
        """
        Classify a hand gesture by matching finger states against detection rules.
        
        This method:
        1. Analyzes current finger states
        2. Compares against all Asamyuta Hasta detection rules
        3. Returns the best matching gesture with confidence score
        
        Args:
            landmarks: List of hand landmarks for one hand.
            hand_label (str): 'Left' or 'Right' hand label.
        
        Returns:
            dict: Classification result with keys:
                - 'gesture': Name of detected gesture
                - 'confidence': Match confidence (0.0 to 1.0)
                - 'description': Gesture description
                - 'finger_states': Current finger state analysis
        """
        # Analyze current finger states
        finger_states = self.analyze_finger_states(landmarks)
        
        # Compare against all Asamyuta gestures
        best_match = None
        best_confidence = 0.0
        
        for gesture in self.gestures.asamyuta_hastas:
            rules = gesture.get('detection_rules', {})
            
            # Calculate match score for this gesture
            match_score = self._calculate_match_score(finger_states, rules, landmarks)
            
            if match_score > best_confidence:
                best_confidence = match_score
                best_match = gesture
        
        # Apply threshold for minimum confidence
        if best_confidence < 0.5:
            return {
                'gesture': 'Unknown',
                'confidence': best_confidence,
                'description': 'No matching gesture detected',
                'finger_states': finger_states
            }
        
        return {
            'gesture': best_match['name'],
            'confidence': best_confidence,
            'description': best_match.get('description', ''),
            'finger_states': finger_states
        }
    
    def _calculate_match_score(self, finger_states: Dict, rules: Dict, landmarks) -> float:
        """
        Calculate how well current finger states match gesture rules.

        Scoring:
        - Direct match = 1.0 point
        - Partial match via similarity = 0.5 points
        - Thumb matches are weighted higher (1.2x) as they're critical for distinction
        - Special rules = 1.0 point
        - Tolerance for angle-based matching to handle detection noise

        Args:
            finger_states: Current detected finger states.
            rules: Expected finger states from gesture definition.
            landmarks: Hand landmarks for special rule checking.

        Returns:
            float: Match score from 0.0 to 1.0 (capped).
        """
        total_checks = 0
        matched_checks = 0

        # Check each finger rule (thumb, index, middle, ring, pinky)
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            if finger in rules:
                total_checks += 1
                expected_state = rules[finger]
                actual_state = finger_states.get(finger, '')

                # Direct match
                if actual_state == expected_state:
                    # Thumb gets higher weight, but normalized
                    weight = 1.2 if finger == 'thumb' else 1.0
                    matched_checks += weight
                # Partial matches for similar states
                elif self._states_are_similar(actual_state, expected_state):
                    weight = 0.6 if finger == 'thumb' else 0.5
                    matched_checks += weight
                # NEW: Tolerance matching - allow nearby states for noisy detection
                elif self._has_tolerance_match(actual_state, expected_state):
                    weight = 0.3  # Lower weight for tolerance matches
                    matched_checks += weight

        # Check special rules (e.g., "index and middle are spread apart")
        if 'special' in rules:
            total_checks += 1
            special_rule = rules['special']
            if self._check_special_rule(special_rule, finger_states, landmarks):
                matched_checks += 1.0

        # CRITICAL FIX: Cap confidence at 1.0 (100%)
        if total_checks == 0:
            return 0.0
        
        raw_score = matched_checks / total_checks
        return min(raw_score, 1.0)  # Cap at 100%

    def _apply_temporal_smoothing(self, hand_key: str, current_gesture: Dict) -> Dict:
        """
        Apply temporal smoothing to stabilize gesture detection.
        
        This prevents flickering between similar gestures by maintaining a history
        and only accepting new gestures if they're stable over multiple frames.
        
        Args:
            hand_key: Unique key for this hand (e.g., 'hand_0', 'hand_1')
            current_gesture: Current gesture detection result
            
        Returns:
            Dict: Smoothed gesture detection result
        """
        # Initialize history for this hand if needed
        if hand_key not in self._gesture_history:
            self._gesture_history[hand_key] = []
        
        # Add current gesture to history
        self._gesture_history[hand_key].append(current_gesture)
        
        # Keep only recent history
        if len(self._gesture_history[hand_key]) > self._smoothing_window:
            self._gesture_history[hand_key] = self._gesture_history[hand_key][-self._smoothing_window:]
        
        # If we don't have enough history, return current gesture
        if len(self._gesture_history[hand_key]) < 3:
            return current_gesture
        
        # Find most common gesture in history
        gesture_counts = {}
        confidence_sums = {}
        
        for gesture_result in self._gesture_history[hand_key]:
            gesture_name = gesture_result.get('gesture', 'Unknown')
            confidence = gesture_result.get('confidence', 0.0)
            
            if gesture_name not in gesture_counts:
                gesture_counts[gesture_name] = 0
                confidence_sums[gesture_name] = 0.0
            
            gesture_counts[gesture_name] += 1
            confidence_sums[gesture_name] += confidence
        
        # Find the dominant gesture
        dominant_gesture = max(gesture_counts.keys(), key=lambda g: gesture_counts[g])
        dominant_count = gesture_counts[dominant_gesture]
        dominant_avg_confidence = confidence_sums[dominant_gesture] / dominant_count
        
        # Only accept new gesture if it appears in majority of recent frames
        if dominant_count >= (self._smoothing_window // 2) + 1:
            # Use the dominant gesture with averaged confidence
            smoothed_result = {
                'gesture': dominant_gesture,
                'confidence': min(dominant_avg_confidence, 1.0),
                'description': current_gesture.get('description', ''),
                'finger_states': current_gesture.get('finger_states', {}),
                'hand_label': current_gesture.get('hand_label', 'Unknown'),
                'smoothed': True
            }
            return smoothed_result
        
        # Otherwise, keep the previous smoothed gesture if available
        return current_gesture

    @staticmethod
    def _has_tolerance_match(actual: str, expected: str) -> bool:
        """
        Check if finger states have tolerance matching (for noisy detection).
        
        This allows for graceful degradation when detection is close but not exact,
        handling the natural variance in MediaPipe hand landmark detection.
        
        Args:
            actual: Detected finger state.
            expected: Expected finger state from rules.
            
        Returns:
            bool: True if states have tolerance match.
        """
        # Define tolerance groups - states that can match with lower confidence
        tolerance_groups = [
            # All straight/bent variants can loosely match
            ['straight', 'straight_spread', 'straight_angled_out', 'stretched_out', 
             'slightly_bent', 'bent', 'bent_forward', 'bent_forward_separated'],
            # All folded variants
            ['folded', 'folded_over_fingers', 'curved'],
            # All touching states
            ['touching_thumb', 'touching_index', 'touching_middle', 'touching_ring', 
             'touching_pinky', 'touching_index_and_middle', 'touching_middle_and_ring',
             'touching_ring_and_pinky', 'touching_middle_ring_pinky', 
             'touching_all_fingertips', 'touching_base_of_ring'],
        ]
        
        for group in tolerance_groups:
            if actual in group and expected in group:
                return True
        return False

    @staticmethod
    def _states_are_similar(actual: str, expected: str) -> bool:
        """
        Check if two finger states are similar enough to be a partial match.

        This allows for graceful degradation when detection is close but not exact.

        Args:
            actual: Detected finger state.
            expected: Expected finger state from rules.

        Returns:
            bool: True if states are similar.
        """
        if actual == expected:
            return True
        
        # Define similarity groups - states that are conceptually similar
        similarity_groups = [
            # Straight variants
            ['straight', 'straight_spread', 'straight_angled_out', 'stretched_out'],
            # Bent variants
            ['bent', 'slightly_bent', 'bent_forward', 'bent_forward_separated'],
            # Folded variants
            ['folded', 'folded_over_fingers'],
            # Curved
            ['curved'],
            # Touching states - thumb touching finger
            ['touching_thumb', 'touching_index', 'touching_middle', 'touching_ring', 'touching_pinky'],
            # Multiple finger touching
            ['touching_index_and_middle', 'touching_middle_and_ring', 'touching_ring_and_pinky', 
             'touching_middle_ring_pinky', 'touching_all_fingertips', 'touching_base_of_ring'],
        ]

        for group in similarity_groups:
            if actual in group and expected in group:
                return True
        return False
    
    def _check_special_rule(self, special_rule: str, finger_states: Dict, landmarks) -> bool:
        """
        Check special detection rules that involve complex finger relationships.

        Args:
            special_rule: Special rule description.
            finger_states: Current finger states.
            landmarks: Hand landmarks.

        Returns:
            bool: True if special rule is satisfied.
        """
        special_rule_lower = special_rule.lower()

        # Check for V-shape (fingers spread apart) - Kartareemukha Hasta
        if 'spread' in special_rule_lower or 'v-shape' in special_rule_lower:
            return finger_states.get('index_middle_spread', False)

        # Check for C-curve - Ardhachandra Hasta
        if 'c curve' in special_rule_lower or 'c-shape' in special_rule_lower:
            # Check if thumb and fingers form a C shape
            thumb_tip = landmarks[THUMB_TIP]
            index_tip = landmarks[INDEX_TIP]
            pinky_tip = landmarks[PINKY_TIP]
            # C-curve: thumb is spread out, fingers grouped
            thumb_index_dist = self._calculate_distance(thumb_tip, index_tip)
            return thumb_index_dist > 0.1 and thumb_index_dist < 0.25

        # Check for cobra hood shape - Sarpasheersha Hasta
        if 'cobra' in special_rule_lower or 'hood' in special_rule_lower:
            # All fingers slightly bent and grouped together
            finger_bent_states = [
                finger_states.get('index'), finger_states.get('middle'),
                finger_states.get('ring'), finger_states.get('pinky')
            ]
            # Allow slightly_bent, bent, or straight (graceful matching)
            return all(state in ['slightly_bent', 'bent', 'straight'] for state in finger_bent_states if state)

        # Check for touching all fingertips - Mukula Hasta, Samdamsha Hasta
        if 'touching' in special_rule_lower and ('tip' in special_rule_lower or 'fingertip' in special_rule_lower):
            # Check thumb touching state
            thumb_state = finger_states.get('thumb', '')
            return 'touching_all' in thumb_state or thumb_state in ['touching_index_and_middle', 'touching_middle_and_ring']

        # Check for "like Pataka but" variations - Hamsapaksha Hasta
        if 'like pataka' in special_rule_lower:
            # Most fingers straight like Pataka
            straight_count = sum(1 for finger in ['index', 'middle', 'ring', 'pinky'] 
                               if 'straight' in str(finger_states.get(finger, '')))
            return straight_count >= 3

        # Check for "lotus" or "flared" pattern - Alapadma Hasta
        if 'lotus' in special_rule_lower or 'flared' in special_rule_lower or 'opened' in special_rule_lower:
            # All fingers straight and spread
            all_straight = all('straight' in str(finger_states.get(f, '')) 
                             for f in ['thumb', 'index', 'middle', 'ring', 'pinky'])
            return all_straight

        # Check for "round ball" or "grasping" - Padmakosha Hasta
        if 'round ball' in special_rule_lower or 'grasping' in special_rule_lower:
            # All fingers curved
            curved_count = sum(1 for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']
                             if finger_states.get(finger) == 'curved')
            return curved_count >= 4

        # Check for "cobra hood shape" - Sarpasheersha
        if 'slowly' in special_rule_lower or 'grouped' in special_rule_lower:
            # Fingers grouped together, slightly bent
            return all(state in ['slightly_bent', 'curved', 'bent'] 
                     for state in [finger_states.get('index'), finger_states.get('middle'),
                                  finger_states.get('ring'), finger_states.get('pinky')] if state)

        # Check for dynamic movement - Samdamsha Hasta
        if 'dynamic' in special_rule_lower or 'repeatedly' in special_rule_lower or 'opening' in special_rule_lower:
            # For static detection, match the closed state (fingertips touching)
            thumb_state = finger_states.get('thumb', '')
            return 'touching' in thumb_state

        # Check for L-shape - Chandrakala Hasta
        if 'l-shape' in special_rule_lower:
            # Thumb and index form L
            thumb_tip = landmarks[THUMB_TIP]
            index_tip = landmarks[INDEX_TIP]
            wrist = landmarks[WRIST]
            # L-shape: thumb stretched out, index straight, perpendicular
            thumb_index_angle = self._calculate_angle(thumb_tip, landmarks[INDEX_MCP], index_tip)
            return 60 < thumb_index_angle < 120

        # Default: try to match based on key terms
        return False
    
    def detect_gestures(self, img, draw=True) -> List[Dict]:
        """
        Detect and classify hand gestures in an image.

        This is the main method that combines hand detection and gesture classification.
        IMPROVED: Now uses temporal smoothing for stable gesture detection.

        Args:
            img (numpy.ndarray): Input image/frame in BGR format.
            draw (bool): If True, draws gesture labels on the image.

        Returns:
            list[dict]: List of classification results for each detected hand.
        """
        # Detect hands
        img = self.find_hands(img, draw=False)

        self.detected_gestures = []

        # Classify gesture for each detected hand
        for idx, landmarks in enumerate(self.hand_landmarks):
            hand_label = self.hand_labels[idx] if idx < len(self.hand_labels) else 'Unknown'

            # Classify the gesture
            result = self.classify_gesture(landmarks, hand_label)
            result['hand_label'] = hand_label
            
            # NEW: Apply temporal smoothing for stable detection
            hand_key = f"hand_{idx}"
            smoothed_result = self._apply_temporal_smoothing(hand_key, result)
            
            self.detected_gestures.append(smoothed_result)

            # Draw gesture label if requested (use smoothed confidence)
            if draw and smoothed_result['confidence'] > 0.5:
                self._draw_gesture_label(img, landmarks, smoothed_result)

        return self.detected_gestures
    
    def reset_gesture_history(self):
        """
        Reset the gesture history buffer.
        
        Call this when starting a new detection session or when you want
        to clear the temporal smoothing buffer.
        """
        self._gesture_history = {}
    
    def _draw_gesture_label(self, img, landmarks, result: Dict):
        """
        Draw gesture classification label on the image.
        
        Args:
            img: Image to draw on.
            landmarks: Hand landmarks.
            result: Classification result dictionary.
        """
        img_height, img_width, _ = img.shape
        
        # Position label above the hand (use wrist or middle finger MCP)
        wrist = landmarks[WRIST]
        middle_mcp = landmarks[MIDDLE_MCP]
        
        # Calculate label position (above the hand)
        label_x = int((wrist.x + middle_mcp.x) / 2 * img_width)
        label_y = int(min(wrist.y, middle_mcp.y) * img_height) - 30
        
        # Prepare label text
        gesture_name = result['gesture']
        confidence = result['confidence']
        label = f"{gesture_name} ({confidence:.0%})"
        
        # Draw background rectangle for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        cv2.rectangle(
            img,
            (label_x - 5, label_y - text_size[1] - 10),
            (label_x + text_size[0] + 5, label_y + 5),
            (0, 0, 0),  # Black background
            -1
        )
        
        # Draw text with color based on confidence
        if confidence > 0.8:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.6:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        cv2.putText(
            img,
            label,
            (label_x, label_y),
            font,
            font_scale,
            color,
            thickness
        )
    
    def get_hands_solution(self):
        """
        Get the MediaPipe Hands solution instance.

        Returns:
            mp.solutions.hands.Hands: Configured Hands solution.
        """
        return self.hands
