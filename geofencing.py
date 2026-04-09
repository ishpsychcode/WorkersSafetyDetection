import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import math
import os

class WorkerSafetySystem:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.hazardous_zone = {'center': (640, 360), 'radius': 150}
        self.alert_active = False
        self.last_alert_time = 0
        self.alert_interval = 3.0
        self.alert_count = 0
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    def point_in_circle(self, point, center, radius):
        distance = math.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
        return distance <= radius
    
    def draw_geofence(self, frame):
        center = self.hazardous_zone['center']
        radius = self.hazardous_zone['radius']
        cv2.circle(frame, center, radius, (0, 0, 255), 3)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.putText(frame, "HAZARDOUS ZONE", (center[0]-100, center[1]-radius-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame
    
    def speak_alert(self):
        """WINDOWS BUILT-IN VOICE - NO INSTALLS!"""
        self.alert_count += 1
        message = "Please step away from hazardous zone"
        print(f"🎤 ALERT #{self.alert_count}: {message}")
        
        # Use Windows PowerShell speech (100% reliable)
        cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak(\\"{message}\\")"'
        os.system(cmd)
    
    def process_frame(self, frame):
        results = self.model(frame, classes=[0])
        in_hazard_zone = False
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    if self.point_in_circle(person_center, self.hazardous_zone['center'],
                                          self.hazardous_zone['radius']):
                        in_hazard_zone = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, person_center, 10, (0, 0, 255), 3)
                        cv2.putText(frame, "🚨 DANGER!", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        self.alert_active = in_hazard_zone
        return frame
    
    def run(self):
        print("🎙️ VOICE ALERT SYSTEM - Every 3 seconds!")
        print("Walk into RED ZONE → SPEAKS 'Please step away...'")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = self.draw_geofence(frame)
            frame = self.process_frame(frame)
            
            # PERFECT TIMING (your beeps proved this works!)
            current_time = time.time()
            time_since_last = current_time - self.last_alert_time
            
            if self.alert_active and time_since_last >= self.alert_interval:
                # SPEAK in background thread
                threading.Thread(target=self.speak_alert, daemon=True).start()
                self.last_alert_time = current_time
            
            # Live display
            next_in = max(0, self.alert_interval - time_since_last)
            status = "🚨 DANGER! Next SPEAK: " + f"{next_in:.1f}s"
            
            color = (0, 0, 255) if self.alert_active else (0, 255, 0)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Voice Alerts: {self.alert_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Worker Safety - VOICE Alerts', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    safety_system = WorkerSafetySystem()
    safety_system.run()