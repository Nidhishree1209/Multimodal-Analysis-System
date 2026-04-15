"""
Run hand gesture detection using webcam.
This is a convenient entry point for the hand gesture detection system.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from demos.hand_gesture_demo import main

if __name__ == "__main__":
    main()
