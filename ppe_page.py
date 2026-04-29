import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import pickle
import tempfile
import os
import time
import threading 
import pandas as pd
from deepface import DeepFace
from scipy.spatial import distance

# --- 1. CONFIGURATION & ENVIRONMENT ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
UDP_URL = "udp://@0.0.0.0:5000?fifo_size=5000000&overrun_nonfatal=1"

# --- 2. CLASS & HELPER DEFINITIONS ---
class VideoStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.ret, self.frame = False, None
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.grab():
                time.sleep(0.01)
                continue
            self.ret, self.frame = self.cap.retrieve()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        time.sleep(0.5)
        if self.cap:
            self.cap.release()

def speak_alert(msg):
    os.system(f'espeak "{msg}"')

@st.cache_resource
def load_models():
    person_model = YOLO("yolov8n.pt")
    ppe_model = YOLO("pbl_1.pt")
    pose_model = YOLO("yolov8n-pose.pt")
    return person_model, ppe_model, pose_model

def is_yellow_dominant(crop):
    if crop.size == 0: return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([35, 255, 255]))
    return (np.count_nonzero(mask) / mask.size) > 0.15

def is_blue_dominant(crop):
    if crop.size == 0: return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([100, 60, 60]), np.array([130, 255, 255]))
    return (np.count_nonzero(mask) / mask.size) > 0.10

# --- 3. LOGIC FUNCTIONS ---
def get_or_create_unknown_id(track_id, current_embedding, UNKNOWN_REUSE_THRESHOLD):
    if track_id in st.session_state.track_to_unknown_id:
        return st.session_state.track_to_unknown_id[track_id]
    best_id, best_dist = None, UNKNOWN_REUSE_THRESHOLD
    for uid, anchor_emb in st.session_state.unknown_embeddings.items():
        dist = distance.cosine(current_embedding, anchor_emb)
        if dist < best_dist:
            best_dist, best_id = dist, uid
    if best_id is not None:
        st.session_state.track_to_unknown_id[track_id] = best_id
        return best_id
    else:
        st.session_state.unknown_counter += 1
        uid = f"Unknown #{st.session_state.unknown_counter}"
        st.session_state.unknown_embeddings[uid] = current_embedding
        st.session_state.track_to_unknown_id[track_id] = uid
        return uid

def identify_worker(frame, x1, y1, x2, y2):
    if (x2 - x1) < 70 or (y2 - y1) < 70: return "Too Far", None
    face_y1, face_y2 = max(0, y1 - int((y2 - y1) * 0.15)), y1 + int((y2 - y1) * 0.45)
    head_crop = frame[face_y1:face_y2, x1:x2]
    if head_crop.size == 0: return "Unknown", None
    try:
        results = DeepFace.represent(head_crop, model_name='Facenet512', enforce_detection=False, detector_backend='opencv')
        if not results: return "Unknown", None
        current_embedding = np.array(results[0]["embedding"])
        current_embedding = current_embedding / np.linalg.norm(current_embedding)
        best_name, min_dist = "Unknown", 0.40
        for saved_item in st.session_state.face_data:
            dist = distance.cosine(current_embedding, saved_item["embedding"])
            if dist < min_dist:
                min_dist, best_name = dist, saved_item["name"]
        return best_name, current_embedding
    except: return "Unknown", None

# --- 4. MAIN PAGE FUNCTION ---
def show_ppe_page():
    if st.button("⬅️ BACK TO MENU"):
        st.session_state['page'] = 'main'
        st.rerun()

    # Initialization of session states
    if 'face_data' not in st.session_state:
        try:
            with open("encodings.pickle", "rb") as f: st.session_state.face_data = pickle.load(f)
        except: st.session_state.face_data = []
    
    states = ['id_memory', 'id_votes', 'last_face_check', 'track_first_seen', 'unknown_embeddings', 'track_to_unknown_id']
    for s in states:
        if s not in st.session_state: st.session_state[s] = {}
    if 'unknown_counter' not in st.session_state: st.session_state.unknown_counter = 0

    person_model, ppe_model, pose_model = load_models()
    REQUIRED_PPE = ["helmet", "vest", "gloves", "glasses"]
    
    # Constants
    ID_CONFIRM_THRESHOLD, UNKNOWN_REUSE_THRESHOLD = 4, 0.35
    FACE_RECHECK_FRAMES, GRACE_PERIOD_FRAMES = 30, 20

    st.sidebar.title(" Optimization Settings")
    mode = st.sidebar.radio("Source:", ["Webcam", "UDP Stream (RPi)", "Upload Video"])
    FRAME_SKIP = st.sidebar.slider("Frame Skip (Inference)", 1, 10, 4)
    PROCESS_WIDTH = 640

    if mode == "UDP Stream (RPi)": source = UDP_URL
    elif mode == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Upload MP4/AVI", type=['mp4', 'avi', 'mov'])
        source = 0
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            source = tfile.name
    else: source = 0

    st.title("PPE & Identity Safety Monitor")
    run = st.checkbox("Start Monitoring", value=False)
    frame_window = st.empty()
    table_placeholder = st.empty()
    fps_display = st.sidebar.empty()

    if run:
        if mode == "UDP Stream (RPi)":
            vs = VideoStream(source).start()
            for _ in range(15): vs.read(); time.sleep(0.05)
            def get_frame(): return vs.read()
            def release_source(): vs.stop()
        else:
            cap = cv2.VideoCapture(source)
            if source == 0:
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            def get_frame(): return cap.read()
            def release_source(): cap.release()

        frame_count, last_results, worker_log, prev_time, last_alert = 0, [], {}, 0, 0

        try:
            while run:
                ret, frame = get_frame()
                if not ret: break
                frame_count += 1
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time
                fps_display.metric("Stream FPS", f"{int(fps)}")

                small_frame = cv2.resize(frame, (PROCESS_WIDTH, int(frame.shape[0] * (PROCESS_WIDTH / frame.shape[1]))))
                scale_x = frame.shape[1] / small_frame.shape[1]

                if frame_count % FRAME_SKIP == 1:
                    results = person_model.track(small_frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=0.5, verbose=False)[0]
                    temp_workers, missing_ppe_names = [], []

                    if results.boxes.id is not None:
                        boxes, ids = results.boxes.xyxy.cpu().numpy(), results.boxes.id.cpu().numpy()
                        for i, box in enumerate(boxes):
                            track_id = int(ids[i])
                            x1, y1, x2, y2 = box.astype(int)
                            if track_id not in st.session_state.track_first_seen: st.session_state.track_first_seen[track_id] = frame_count
                            frames_alive = frame_count - st.session_state.track_first_seen[track_id]

                            if frames_alive < GRACE_PERIOD_FRAMES:
                                temp_workers.append({'name': "...", 'bbox': (x1, y1, x2, y2), 'missing': [], 'ppe_status': {item: "N/A" for item in REQUIRED_PPE}, 'in_grace': True})
                                continue

                            # Identity
                            already_confirmed = track_id in st.session_state.id_memory
                            frames_since_check = frame_count - st.session_state.last_face_check.get(track_id, -9999)

                            if already_confirmed: name = st.session_state.id_memory[track_id]
                            elif frames_since_check < FACE_RECHECK_FRAMES:
                                votes = st.session_state.id_votes.get(track_id, {})
                                name = max(votes, key=votes.get) if votes else "Identifying..."
                            else:
                                st.session_state.last_face_check[track_id] = frame_count
                                raw_name, embedding = identify_worker(frame, int(x1*scale_x), int(y1*scale_x), int(x2*scale_x), int(y2*scale_x))
                                if raw_name not in ["Unknown", "Too Far"] and embedding is not None:
                                    votes = st.session_state.id_votes.setdefault(track_id, {})
                                    votes[raw_name] = votes.get(raw_name, 0) + 1
                                    if votes[raw_name] >= ID_CONFIRM_THRESHOLD:
                                        st.session_state.id_memory[track_id] = raw_name
                                        name = raw_name
                                    else: name = "Identifying..."
                                elif raw_name == "Unknown" and embedding is not None:
                                    name = get_or_create_unknown_id(track_id, embedding, UNKNOWN_REUSE_THRESHOLD)
                                else: name = raw_name

                            # PPE Detection
                            person_crop = small_frame[y1:y2, x1:x2]
                            detected_ppe, face_visible, hands_visible, gloves_color_detected, glasses_color_detected = [], False, False, False, False

                            if person_crop.size > 0:
                                pose_res = pose_model(person_crop, verbose=False)[0]
                                if pose_res.keypoints is not None and pose_res.keypoints.conf is not None:
                                    if pose_res.keypoints.conf.shape[0] > 0:
                                        kpts_conf, kpts_xy = pose_res.keypoints.conf[0].cpu().numpy(), pose_res.keypoints.xy[0].cpu().numpy()
                                        if len(kpts_conf) >= 17:
                                            if np.sum(kpts_conf[0:5] > 0.5) >= 3: face_visible = True
                                            if kpts_conf[9] > 0.5 or kpts_conf[10] > 0.5: hands_visible = True
                                            for wrist_idx in [9, 10]:
                                                if kpts_conf[wrist_idx] > 0.5:
                                                    wx, wy, pad = int(kpts_xy[wrist_idx][0]), int(kpts_xy[wrist_idx][1]), 20
                                                    wrist_crop = person_crop[max(0, wy-pad):min(person_crop.shape[0], wy+pad), max(0, wx-pad):min(person_crop.shape[1], wx+pad)]
                                                    if is_yellow_dominant(wrist_crop): gloves_color_detected = True; break
                                            if kpts_conf[0] > 0.5 and (kpts_conf[1] > 0.5 or kpts_conf[2] > 0.5):
                                                eye_points = [kpts_xy[0]]
                                                if kpts_conf[1] > 0.5: eye_points.append(kpts_xy[1])
                                                if kpts_conf[2] > 0.5: eye_points.append(kpts_xy[2])
                                                all_pts = np.array(eye_points)
                                                ex1, ey1 = max(0, int(np.min(all_pts[:, 0])) - 15), max(0, int(np.min(all_pts[:, 1])) - 15)
                                                ex2, ey2 = min(person_crop.shape[1], int(np.max(all_pts[:, 0])) + 15), min(person_crop.shape[0], int(np.max(all_pts[:, 1])) + 15)
                                                if is_blue_dominant(person_crop[ey1:ey2, ex1:ex2]): glasses_color_detected = True

                                ppe_res = ppe_model(person_crop, verbose=False)[0]
                                for p_box in ppe_res.boxes:
                                    p_cls, p_conf, p_label = int(p_box.cls[0]), float(p_box.conf[0]), ppe_model.names[int(p_box.cls[0])].lower()
                                    if p_label == "gloves" and p_conf < 0.15: continue
                                    if p_label == "glasses" and p_conf < 0.20: continue
                                    if p_label in ["helmet", "vest"] and p_conf < 0.4: continue
                                    detected_ppe.append(p_label)

                            # Status Logic
                            status_report, missing = {}, []
                            for item in REQUIRED_PPE:
                                if item == "glasses" and not face_visible: status_report[item] = "N/A"
                                elif item == "gloves" and not hands_visible: status_report[item] = "N/A"
                                elif item == "gloves":
                                    if "gloves" in detected_ppe or gloves_color_detected: status_report[item] = "Yes"
                                    else: status_report[item] = "No"; missing.append(item)
                                elif item == "glasses":
                                    if "glasses" in detected_ppe or glasses_color_detected: status_report[item] = "Yes"
                                    else: status_report[item] = "No"; missing.append(item)
                                elif item in detected_ppe: status_report[item] = "Yes"
                                else: status_report[item] = "No"; missing.append(item)

                            if name not in ["Identifying...", "..."]: worker_log[name] = status_report
                            if missing and name not in ["Identifying...", "...", "Too Far"]: missing_ppe_names.append((name, missing))
                            temp_workers.append({'name': name, 'bbox': (x1, y1, x2, y2), 'missing': missing, 'ppe_status': status_report, 'in_grace': False})
                    last_results = temp_workers

                    if missing_ppe_names and (curr_time - last_alert > 8):
                        n_str, itms = missing_ppe_names[0]
                        threading.Thread(target=speak_alert, args=(f"{n_str} is missing {', '.join(itms)}",), daemon=True).start()
                        last_alert = curr_time

                for worker in last_results:
                    rx1, ry1, rx2, ry2 = worker['bbox']
                    name, s = worker['name'], worker['ppe_status']
                    if worker.get('in_grace') or name in ["...", "Identifying..."]:
                        cv2.rectangle(small_frame, (rx1, ry1), (rx2, ry2), (200, 200, 200), 2)
                        cv2.putText(small_frame, name, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                        continue
                    color = (0, 255, 0) if not worker['missing'] else (0, 0, 255)
                    if "Unknown" in name: color = (0, 255, 255)
                    label = f"{name} | H:{s.get('helmet','?')} V:{s.get('vest','?')} G:{s.get('gloves','?')} Gl:{s.get('glasses','?')}"
                    cv2.rectangle(small_frame, (rx1, ry1), (rx2, ry2), color, 2)
                    cv2.putText(small_frame, label, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                frame_window.image(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                if worker_log:
                    df = pd.DataFrame([{"Worker": k, **v} for k, v in worker_log.items()])
                    table_placeholder.dataframe(df, use_container_width=True, height=200)
                time.sleep(0.01)
        finally:
            release_source()