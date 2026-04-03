import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import pickle
import tempfile
import os
import time
from deepface import DeepFace
from scipy.spatial import distance

# -------------------------
# FIX WINDOWS / PERFORMANCE ENV
# -------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# -------------------------
# SESSION STATE
# -------------------------
if 'face_data' not in st.session_state:
    try:
        with open("encodings.pickle", "rb") as f:
            st.session_state.face_data = pickle.load(f)
    except:
        st.error("⚠️ encodings.pickle not found!")
        st.session_state.face_data = []

if 'id_memory' not in st.session_state:
    st.session_state.id_memory = {}

# -------------------------
# LOAD MODELS (Cached)
# -------------------------
@st.cache_resource
def load_models():
    # Use 'yolov8n.pt' for maximum speed
    person_model = YOLO("yolov8n.pt") 
    ppe_model = YOLO("pbl_1.pt")
    return person_model, ppe_model

person_model, ppe_model = load_models()

# -------------------------
# FACE RECOGNITION (Optimized)
# -------------------------
def identify_worker(frame, x1, y1, x2, y2):
    # Skip small detections to save compute
    if (x2 - x1) < 70 or (y2 - y1) < 70:
        return "Too Far"

    # Precise Head Crop
    face_y1 = max(0, y1 - int((y2 - y1) * 0.15))
    face_y2 = y1 + int((y2 - y1) * 0.45)
    head_crop = frame[face_y1:face_y2, x1:x2]

    if head_crop.size == 0:
        return "Unknown"

    try:
        # Using Facenet512 - ensure the model is downloaded
        results = DeepFace.represent(
            head_crop,
            model_name='Facenet512',
            enforce_detection=False,
            detector_backend='opencv' # Faster than mtcnn for real-time
        )

        if not results:
            return "Unknown"

        current_embedding = np.array(results[0]["embedding"])
        current_embedding = current_embedding / np.linalg.norm(current_embedding)

        best_name = "Unknown"
        min_dist = 0.5  # Slightly relaxed threshold for movement

        for saved_item in st.session_state.face_data:
            dist = distance.cosine(current_embedding, saved_item["embedding"])
            if dist < min_dist:
                min_dist = dist
                best_name = saved_item["name"]

        return best_name
    except:
        return "Unknown"

# -------------------------
# UI SETUP
# -------------------------
st.set_page_config(page_title="AI Safety Monitor", layout="wide")
st.sidebar.title("🚀 Optimization Settings")
mode = st.sidebar.radio("Source:", ["Webcam", "Upload Video"])

# Real-time Tuning
FRAME_SKIP = st.sidebar.slider("Frame Skip (Inference)", 1, 10, 4)
PROCESS_WIDTH = 640 # Internal processing resolution

source = 0
if mode == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload MP4/AVI", type=['mp4', 'avi', 'mov'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        source = tfile.name

st.title("👷 PPE & Identity Safety Monitor")
run = st.checkbox("Start Monitoring", value=False)
frame_window = st.empty()
fps_display = st.sidebar.empty()

# -------------------------
# MAIN EXECUTION
# -------------------------
if run:
    cap = cv2.VideoCapture(source)
    if source == 0:
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    last_results = []
    last_ppe_boxes = []
    prev_time = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            st.info("Stream ended or file finished.")
            break

        frame_count += 1
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        fps_display.metric("Stream FPS", f"{int(fps)}")

        # 1. Resize for YOLO speed (Don't process high-res)
        small_frame = cv2.resize(frame, (PROCESS_WIDTH, int(frame.shape[0] * (PROCESS_WIDTH / frame.shape[1]))))
        scale_x = frame.shape[1] / small_frame.shape[1]
        scale_y = frame.shape[0] / small_frame.shape[1] # Note: use small_frame.shape[0] normally

        # 2. YOLO Inference Block
        if frame_count % FRAME_SKIP == 1:
            # Person Tracking
            results = person_model.track(
                small_frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=0.5,
                verbose=False
            )[0]

            # PPE Detection (runs every 2nd inference cycle)
            if frame_count % (FRAME_SKIP * 2) == 1:
                ppe_results = ppe_model(small_frame, conf=0.5, verbose=False)[0]
                last_ppe_boxes = [(int(p.cls[0]), p.xyxy[0].cpu().numpy()) for p in ppe_results.boxes]

            temp_workers = []
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                ids = results.boxes.id.cpu().numpy()

                for i, box in enumerate(boxes):
                    track_id = int(ids[i])
                    x1, y1, x2, y2 = box.astype(int)

                    # Identity Logic
                    if track_id in st.session_state.id_memory:
                        name = st.session_state.id_memory[track_id]
                    else:
                        # Use high-res crop for better face recognition
                        # Scale back the coordinates to original frame size
                        orig_x1, orig_y1 = int(x1 * scale_x), int(y1 * scale_x)
                        orig_x2, orig_y2 = int(x2 * scale_x), int(y2 * scale_x)
                        
                        name = identify_worker(frame, orig_x1, orig_y1, orig_x2, orig_y2)
                        if name not in ["Unknown", "Too Far"]:
                            st.session_state.id_memory[track_id] = name

                    # PPE Overlap Check (using small_frame coordinates)
                    h, v, g = False, False, False
                    for cls, p_box in last_ppe_boxes:
                        cx, cy = (p_box[0] + p_box[2]) / 2, (p_box[1] + p_box[3]) / 2
                        if x1 < cx < x2 and y1 < cy < y2:
                            if cls == 0: h = True
                            elif cls == 1: v = True
                            elif cls == 2: g = True

                    temp_workers.append({'name': name, 'bbox': (x1, y1, x2, y2), 'ppe': (h, v, g)})
            
            last_results = temp_workers

        # 3. Drawing on the SMALL frame (Faster than drawing on 1080p)
        for worker in last_results:
            rx1, ry1, rx2, ry2 = worker['bbox']
            h, v, g = worker['ppe']
            name = worker['name']

            # Color Coding
            if name != "Unknown" and h and v:
                color = (0, 255, 0) # Safe
            elif name == "Unknown":
                color = (0, 255, 255) # Warning
            else:
                color = (0, 0, 255) # Danger

            cv2.rectangle(small_frame, (rx1, ry1), (rx2, ry2), color, 2)
            cv2.putText(small_frame, f"{name}", (rx1, ry1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Status text
            status = f"H:{int(h)} V:{int(v)}"
            cv2.putText(small_frame, status, (rx1, ry2 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 4. Optimized Display
        frame_window.image(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Brief sleep to prevent Streamlit UI thread locking
        time.sleep(0.01)

    cap.release()
    st.write("Stopped monitoring.")