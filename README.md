#  Worker Safety & Compliance Monitoring System

##  Overview
This project is an AI-powered worker safety and productivity monitoring system designed for hazardous environments. It ensures workers follow safety protocols, remain within designated zones, and maintain proper working posture using advanced computer vision techniques.

---

##  Key Features

###  Safety Compliance Detection
- Detects whether workers are wearing required safety gear (helmet, vest, etc.)

###  Geofencing (Hazard Zone Monitoring)
- Tracks whether workers are inside or outside predefined hazardous zones
- Alerts for unauthorized access or unsafe positioning

###  Pose Estimation-Based Analysis
- Monitors worker posture and movement
- Detects unsafe bending positions
- Enables productivity tracking based on activity

###  Object Tracking
- Tracks individual workers across frames
- Maintains identity for continuous monitoring

###  Fall Detection
- Identifies fall incidents in real-time
- Can be extended to trigger alerts

---

##  Tech Stack
- Python
- OpenCV
- YOLO (Ultralytics)
- Pose Estimation Models (MediaPipe / OpenPose)
- NumPy

---
## Custom Trained Model

This is the link for the custom trained model we used 

-> https://drive.google.com/file/d/1gP6rbCbWnnN2JzCHyBeIaomKCnwh3Bln/view?usp=sharing
---
##  Project Structure

```
WorkersSafetyDetection/
│
├── src/                    # Core implementation
│   ├── detect.py
│   ├── tracking.py
│   ├── pose_estimation.py
│   └── geofencing.py
│
├── models/                 # Trained models
├── data/                   # Datasets
│
├── README.md
├── requirements.txt
```

---

## How to Run

1. Install dependencies:

```
   pip install -r requirements.txt
```

3. Run the system:
```
   python src/detect.py
```



