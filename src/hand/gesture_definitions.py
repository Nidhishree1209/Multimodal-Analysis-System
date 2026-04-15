"""
Bharathanatyam Hand Gesture Definitions
========================================
This module loads and manages hand gesture definitions from the 
Bharathanatyam datasets, including both Asamyuta (single hand) and 
Samyuta (combined hands) hastas.

MediaPipe Hand Landmarks (21 points):
- 0: Wrist
- 1-4: Thumb (CMC, MCP, IP, TIP)
- 5-8: Index finger (MCP, PIP, DIP, TIP)
- 9-12: Middle finger (MCP, PIP, DIP, TIP)
- 13-16: Ring finger (MCP, PIP, DIP, TIP)
- 17-20: Pinky finger (MCP, PIP, DIP, TIP)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


# Landmark index constants for MediaPipe Hand landmarks
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


class BharathanatyamGestures:
    """
    Manages Bharathanatyam hand gesture definitions from dataset files.
    
    Attributes:
        asamyuta_hastas (list): Single hand gestures (28 types)
        samyuta_hastas (list): Combined hand gestures (24 types)
        all_gestures (dict): Combined dictionary of all gestures by name
    """
    
    def __init__(self, dataset_dir: Optional[str] = None):
        """
        Initialize gesture definitions by loading from JSON files.
        
        Args:
            dataset_dir (str, optional): Path to dataset directory.
                If None, uses default location relative to this file.
        """
        if dataset_dir is None:
            # Default to bharathanatyam_datasets in project root
            self.dataset_dir = Path(__file__).parent.parent.parent / "bharathanatyam_datasets"
        else:
            self.dataset_dir = Path(dataset_dir)
        
        self.asamyuta_hastas = []
        self.samyuta_hastas = []
        self.all_gestures = {}
        
        # Load gesture definitions
        self._load_gestures()
    
    def _load_gestures(self):
        """Load all gesture definitions from JSON files."""
        # Load Asamyuta Hastas (single hand gestures)
        asamyuta_file = self.dataset_dir / "asamyuta_hastas.json"
        if asamyuta_file.exists():
            with open(asamyuta_file, 'r', encoding='utf-8') as f:
                self.asamyuta_hastas = json.load(f)
            print(f"✓ Loaded {len(self.asamyuta_hastas)} Asamyuta Hastas")
        else:
            print(f"⚠ Warning: {asamyuta_file} not found")
        
        # Load Samyuta Hastas (combined hand gestures)
        samyuta_file = self.dataset_dir / "samyuta_hastas.json"
        if samyuta_file.exists():
            with open(samyuta_file, 'r', encoding='utf-8') as f:
                self.samyuta_hastas = json.load(f)
            print(f"✓ Loaded {len(self.samyuta_hastas)} Samyuta Hastas")
        else:
            print(f"⚠ Warning: {samyuta_file} not found")
        
        # Build combined gesture dictionary
        for gesture in self.asamyuta_hastas + self.samyuta_hastas:
            name = gesture['name']
            self.all_gestures[name] = gesture
    
    def get_asamyuta_gesture(self, index: int) -> Optional[Dict]:
        """
        Get an Asamyuta (single hand) gesture by index.
        
        Args:
            index (int): Index of the gesture (0-27).
            
        Returns:
            dict or None: Gesture definition or None if index is invalid.
        """
        if 0 <= index < len(self.asamyuta_hastas):
            return self.asamyuta_hastas[index]
        return None
    
    def get_samyuta_gesture(self, index: int) -> Optional[Dict]:
        """
        Get a Samyuta (combined hand) gesture by index.
        
        Args:
            index (int): Index of the gesture (0-23).
            
        Returns:
            dict or None: Gesture definition or None if index is invalid.
        """
        if 0 <= index < len(self.samyuta_hastas):
            return self.samyuta_hastas[index]
        return None
    
    def get_gesture_by_name(self, name: str) -> Optional[Dict]:
        """
        Get a gesture definition by name.
        
        Args:
            name (str): Name of the gesture (e.g., "Pataka Hasta").
            
        Returns:
            dict or None: Gesture definition or None if not found.
        """
        return self.all_gestures.get(name)
    
    def get_all_asamyuta_names(self) -> List[str]:
        """Get list of all Asamyuta Hasta names."""
        return [g['name'] for g in self.asamyuta_hastas]
    
    def get_all_samyuta_names(self) -> List[str]:
        """Get list of all Samyuta Hasta names."""
        return [g['name'] for g in self.samyuta_hastas]
    
    def get_finger_rules(self, gesture_name: str) -> Optional[Dict]:
        """
        Get the finger detection rules for a specific gesture.
        
        Args:
            gesture_name (str): Name of the gesture.
            
        Returns:
            dict or None: Detection rules or None if gesture not found.
            
        Example:
            rules = gestures.get_finger_rules("Pataka Hasta")
            # Returns: {'thumb': 'straight', 'index': 'straight', ...}
        """
        gesture = self.get_gesture_by_name(gesture_name)
        if gesture:
            return gesture.get('detection_rules')
        return None
    
    def get_gesture_info_text(self, gesture_name: str) -> str:
        """
        Get formatted information text for a gesture.
        
        Args:
            gesture_name (str): Name of the gesture.
            
        Returns:
            str: Formatted text with description and meanings.
        """
        gesture = self.get_gesture_by_name(gesture_name)
        if not gesture:
            return "Gesture not found"
        
        info = f"Gesture: {gesture['name']}\n"
        info += f"Description: {gesture['description']}\n\n"
        info += "Meanings/Uses:\n"
        
        # Parse viniyoga_translation to extract individual meanings
        translation = gesture.get('viniyoga_translation', '')
        if translation:
            # Split by comma and format each meaning
            meanings = [m.strip() for m in translation.split(',') if m.strip()]
            for i, meaning in enumerate(meanings[:10], 1):  # Show first 10 meanings
                info += f"  {i}. {meaning}\n"
            if len(meanings) > 10:
                info += f"  ... and {len(meanings) - 10} more"
        
        return info
    
    def is_samyuta_gesture(self, gesture_name: str) -> bool:
        """
        Check if a gesture is a Samyuta (requires both hands) gesture.
        
        Args:
            gesture_name (str): Name of the gesture.
            
        Returns:
            bool: True if gesture requires both hands.
        """
        for gesture in self.samyuta_hastas:
            if gesture['name'] == gesture_name:
                return True
        return False
    
    def get_total_gesture_count(self) -> int:
        """Get total number of gestures loaded."""
        return len(self.asamyuta_hastas) + len(self.samyuta_hastas)
