import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# Load models (lightweight)
pose_model = YOLO("yolov8n-pose.pt")
phone_model = YOLO("yolov8n.pt")

# Pi Camera setup (LOW resolution for speed)
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (480, 360)}
))
picam2.start()

# Tracking
worker_id_map = {}
next_worker_id = 1

def get_worker_id(x, y):
    global next_worker_id
    for wid, (px, py) in worker_id_map.items():
        if abs(x - px) < 50 and abs(y - py) < 50:
            worker_id_map[wid] = (x, y)
            return wid
    worker_id_map[next_worker_id] = (x, y)
    next_worker_id += 1
    return next_worker_id - 1

frame_count = 0
phones = []

while True:

    frame = picam2.capture_array()

    # Mirror
    frame = cv2.flip(frame, 1)

    # Pose detection (every frame)
    pose_results = pose_model(frame, conf=0.4, verbose=False)

    # 🔥 Phone detection only every 5 frames (BOOST FPS)
    if frame_count % 5 == 0:
        phone_results = phone_model(frame, conf=0.3, verbose=False)
        phones = []

        for r in phone_results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                if int(cls) == 67:  # phone
                    phones.append(box.cpu().numpy())

    frame_count += 1

    if pose_results[0].keypoints is not None:

        keypoints = pose_results[0].keypoints.xy.cpu().numpy()

        for person in keypoints:

            # Keypoints
            nose = person[0]
            l_shoulder = person[5]
            r_shoulder = person[6]
            l_hip = person[11]
            r_hip = person[12]
            l_wrist = person[9]
            r_wrist = person[10]

            # Center
            cx = int((l_hip[0] + r_hip[0]) / 2)
            cy = int((l_hip[1] + r_hip[1]) / 2)

            worker_id = get_worker_id(cx, cy)

            # ===== BENDING =====
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            hip_y = (l_hip[1] + r_hip[1]) / 2
            hip_x = (l_hip[0] + r_hip[0]) / 2

            vertical_drop = abs(shoulder_y - hip_y) < 80
            forward_lean = abs(nose[0] - hip_x) > 40

            # ===== PHONE =====
            phone_near = False

            for (x1, y1, x2, y2) in phones:

                phone_cx = (x1 + x2) / 2
                phone_cy = (y1 + y2) / 2

                dist_left = np.linalg.norm([phone_cx - l_wrist[0],
                                           phone_cy - l_wrist[1]])

                dist_right = np.linalg.norm([phone_cx - r_wrist[0],
                                            phone_cy - r_wrist[1]])

                dist_head = np.linalg.norm([phone_cx - nose[0],
                                           phone_cy - nose[1]])

                if dist_left < 70 or dist_right < 70:
                    phone_near = True

                elif dist_head < 70:
                    phone_near = True

            # ===== ACTION =====
            action = "IDLE"

            if nose[1] > hip_y:
                action = "FALL DETECTED"

            elif vertical_drop and forward_lean:
                action = "BENDING"

            elif phone_near:
                action = "PHONE USAGE"

            else:
                action = "IDLE"

            label = f"W{worker_id}: {action}"

            cv2.putText(frame, label,
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2)

    # Draw skeleton
    annotated = pose_results[0].plot()

    cv2.imshow("Worker Safety Monitor (Raspberry Pi)", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()