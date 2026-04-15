# Multimodal Analysis System

> Internship Project - NIDHISHREE N

## Overview
A multimodal analysis system that combines **pose detection**, **facial expression recognition**, and **gesture analysis** with **Explainable AI (XAI)** for interpretability.

## Focus Areas
- **Text + Video + Pose Analysis Integration**
- **MediaPipe** for pose, face, and hand detection
- **XAI Techniques** (SHAP, LIME) for model interpretability
- **Behavioral Intelligence** through gesture recognition

## Tech Stack
- **Backend**: Python 3.11, MediaPipe, OpenCV, Scikit-Learn
- **XAI**: SHAP, LIME
- **Frontend**: React (Node.js) - *Days 8-9*
- **Database**: PostgreSQL / MongoDB - *Days 8-9*

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running Demos

```bash
# Pose Detection Demo
python demos/pose_detection_demo.py

# Hand Gesture Detection (Bharathanatyam Hastas)
python demos/hand_gesture_demo.py

# Or use the convenient wrapper script
python run_webcam.py
```

## Project Structure

```
multimodal-analysis-system/
├── src/                    # Main source code
│   ├── pose/               # Pose detection modules
│   ├── face/               # Face detection modules
│   └── hand/               # Hand detection modules
├── demos/                  # Runnable demo scripts
├── data/                   # Saved data & models
├── tests/                  # Test files
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Internship Timeline

| Days | Component | Deliverable |
|------|-----------|-------------|
| 1-2 | Environment Setup & MediaPipe Basics | Pose detection demo |
| 3-4 | Pose & Gesture Recognition | Pose estimation module |
| 5-6 | Face Mesh & Expressions | Face expression module |
| 7 | XAI Implementation | Interpretability report |
| 8-9 | Full-Stack Integration | Integrated system |
| 9-10 | Advanced Analytics & Deployment | Full system with analytics |

## License
MIT
