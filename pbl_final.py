import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import os
import json
from collections import deque
from ppe_page import show_ppe_page

# --- 1. PAGE CONFIG & GLOBAL STYLING ---
st.set_page_config(layout="wide", page_title="RPi AI Workplace Suite", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main .block-container { padding-top: 1rem; height: 100vh; overflow: hidden; }
    .stImage > img { max-height: 58vh; width: auto; margin: auto; display: block; border-radius: 8px; border: 2px solid #333; }
    .stButton>button { 
        width: 100%; border-radius: 0px; background-color: #ffffff; 
        color: #000; font-weight: 700; height: 3.5rem; text-transform: uppercase; margin-bottom: 5px; 
    }
    .status-box { background-color: #111; padding: 30px 20px; border-left: 10px solid #444; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state['page'] = 'main'

# --- CONFIGURATION ---
UDP_URL = "udp://@0.0.0.0:5000?fifo_size=5000000&overrun_nonfatal=1"
GEOFENCE_FILE = "geofence_config.json"
ROI_FILE = "roi_points.json"

# --- 2. THREADED VIDEO ENGINE ---
class VideoStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.ret, self.frame = False, None
        self.stopped = False
        
    def start(self):
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        if self.cap: self.cap.release()

def speak_alert(msg):
    cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak(\\"{msg}\\")"'
    os.system(cmd)

# --- 3. PAGE: MAIN MENU ---
def show_main_menu():
    st.title("RPI MONITORING HUB")
    col1, col2, col3 = st.columns(3) # Changed from 2 to 3
    with col1: # Added new column
        st.info("Module 1 : PPE Compliance and Identity Tracking")
        if st.button("PPE DETECTION"):
            st.session_state['page'] = 'ppe'
            st.rerun()
    with col2:
        st.info("Module 2 : Productivity Tracking")
        if st.button("PRODUCTIVITY"):
            st.session_state['page'] = 'productivity'
            st.rerun()
    with col3:
        st.info("Module 3 : Safety Geofence & Fall Detection")
        if st.button("GEOFENCE & FALL"):
            st.session_state['page'] = 'geofence'
            st.rerun()
    

# --- 4. PAGE: GEOFENCE & FALL (MODIFIED WITH ULTRASENSITIVE LOGIC) ---
def show_geofence_page():
    if st.button("⬅️ BACK TO MENU"):
        st.session_state['page'] = 'main'
        st.rerun()

    st.title("GEOFENCE & FALL MONITOR")
    col_video, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown("""
        - **Drag Circles**: Adjust Zone
        - **'p'**: Clear & Redraw
        - **Right Click**: Save Redraw
        """)
        run_monitor = st.checkbox("START MONITORING", key="geo_check")
        status_disp = st.empty()
    
    with col_video:
        frame_window = st.empty()

    if run_monitor:
        # State Initialization from UltraSensitive Class
        state = {
            "poly": np.array([[200,200], [400,200], [400,400], [200,400]], np.int32),
            "drawing": False, "clicks": [], "selected": None
        }
        
        # Fall Detection Constants
        FALL_ANGLE_THRESHOLD = 40
        FALL_SUSTAIN_FRAMES = 8
        FALL_VELOCITY_THRESHOLD = 6
        FALL_DISPLAY_DURATION = 3.0

        if os.path.exists(GEOFENCE_FILE):
            try:
                with open(GEOFENCE_FILE, "r") as f:
                    state["poly"] = np.array(json.load(f), dtype=np.int32)
            except: pass

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if state["drawing"]:
                    state["clicks"].append([x, y])
                else:
                    for i, pt in enumerate(state["poly"]):
                        if abs(pt[0]-x) < 20 and abs(pt[1]-y) < 20: 
                            state["selected"] = i
            elif event == cv2.EVENT_MOUSEMOVE:
                if state["selected"] is not None:
                    state["poly"][state["selected"]] = [x, y]
            elif event == cv2.EVENT_LBUTTONUP:
                state["selected"] = None
                with open(GEOFENCE_FILE, "w") as f: json.dump(state["poly"].tolist(), f)
            elif event == cv2.EVENT_RBUTTONDOWN and state["drawing"] and len(state["clicks"]) >= 3:
                state["poly"] = np.array(state["clicks"], np.int32)
                state["drawing"] = False
                with open(GEOFENCE_FILE, "w") as f: json.dump(state["poly"].tolist(), f)

      
        WINDOW_NAME = "Geofence ROI"

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, mouse_cb)
        
        vs = VideoStream(UDP_URL).start()
        model = YOLO('yolov8n-pose.pt')
        
        # Fall tracking history
        person_history = {} # pid: deque
        low_angle_count = {} # pid: int
        fall_confirmed = {} # pid: timestamp
        last_alert_time = 0

        try:
            while True:
                if not st.session_state.get("geo_check", False):
                    break

                ret, frame = vs.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                now = time.time()
                poly_reshaped = state["poly"].reshape((-1, 1, 2))
                
                # Draw Danger Zone
                overlay = frame.copy()
                cv2.fillPoly(overlay, [poly_reshaped], (0, 0, 255))
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.polylines(frame, [poly_reshaped], True, (0, 0, 255), 3)
                for pt in state["poly"]:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (255, 255, 0), -1)

                results = model.predict(frame, verbose=False)
                danger_detected = False
                fall_detected = False

                for r in results:
                    if r.keypoints is None: continue
                    kpts = r.keypoints.xy.cpu().numpy()

                    for pid, person in enumerate(kpts):
                        if np.sum(person) == 0: continue
                        
                        # Nose for text placement
                        nose = person[0]
                        
                        # --- 1. FOOT-ONLY GEOFENCE ---
                        for f_idx in [15, 16]: # ankles
                            foot = person[f_idx]
                            fx, fy = int(foot[0]), int(foot[1])
                            if fx > 0 and cv2.pointPolygonTest(poly_reshaped, (float(fx), float(fy)), False) >= 0:
                                danger_detected = True
                                cv2.circle(frame, (fx, fy), 8, (0, 0, 255), -1)
                                cv2.putText(frame, "FOOT IN ZONE", (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                        # --- 2. FALL DETECTION (ANGLE + VELOCITY) ---
                        ls, rs, lh, rh = person[5], person[6], person[11], person[12]
                        if all(kp[0] > 0 for kp in [ls, rs, lh, rh]):
                            # Calculate Torso Angle
                            mid_s = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
                            mid_h = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
                            dx, dy = mid_s[0] - mid_h[0], mid_s[1] - mid_h[1]
                            angle = np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-6))
                            hip_y = (lh[1] + rh[1]) / 2.0

                            # Track History
                            if pid not in person_history:
                                person_history[pid] = deque(maxlen=15)
                                low_angle_count[pid] = 0
                            
                            hist = person_history[pid]
                            hist.append((hip_y, angle))

                            # Angle Logic
                            if angle < FALL_ANGLE_THRESHOLD:
                                low_angle_count[pid] += 1
                            else:
                                low_angle_count[pid] = 0
                            
                            # Velocity Logic
                            velocity_condition = False
                            if len(hist) >= 3:
                                ys = [h[0] for h in hist]
                                max_delta = max(ys[i+1] - ys[i] for i in range(len(ys)-1))
                                velocity_condition = max_delta >= FALL_VELOCITY_THRESHOLD

                            if low_angle_count[pid] >= FALL_SUSTAIN_FRAMES and velocity_condition:
                                fall_confirmed[pid] = now

                            # Display live angle
                            nx, ny = int(nose[0]), int(nose[1])
                            if nx > 0:
                                cv2.putText(frame, f"Torso: {angle:.0f}deg", (nx, ny - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

                        # Check persistent fall display
                        if pid in fall_confirmed:
                            if now - fall_confirmed[pid] < FALL_DISPLAY_DURATION:
                                fall_detected = True
                            else:
                                del fall_confirmed[pid]

                        # Draw Keypoints
                        for x, y in person:
                            cv2.circle(frame, (int(x), int(y)), 3, (0,255,0), -1)

                # --- 3. UI & ALERTS ---
                color, status = "#00FF88", "CLEAR"
                if fall_detected:
                    color, status = "#FF0000", "FALL DETECTED!"
                    if now - last_alert_time > 3.0:
                        threading.Thread(target=speak_alert, args=("FALL DETECTED",), daemon=True).start()
                        last_alert_time = now
                    cv2.putText(frame, "FALL DETECTED!", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                elif danger_detected:
                    color, status = "#FF0000", "INTRUSION"
                    if now - last_alert_time > 3.0:
                        threading.Thread(target=speak_alert, args=("Please step away from hazardous zone",), daemon=True).start()
                        last_alert_time = now

                status_disp.markdown(f"<div class='status-box' style='border-left-color:{color};'><h1 style='color:{color};'>{status}</h1></div>", unsafe_allow_html=True)
                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cv2.imshow(WINDOW_NAME, frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('p'): state["drawing"], state["clicks"] = True, []
        finally:
            vs.stop()
            cv2.destroyAllWindows()

# --- 5. PRODUCTIVITY (UNCHANGED) ---
# --- 5. PRODUCTIVITY (REFACTORED WITH INTERACTIVE ROI) ---

def get_roi_interactively(stream_url):
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    for _ in range(10): cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret: return None
    points = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(img_copy, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
            cv2.imshow("Select ROI then press Enter", img_copy)
    img_copy = frame.copy()
    cv2.imshow("Select ROI then press Enter", img_copy)
    cv2.setMouseCallback("Select ROI then press Enter", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(points) >= 3:
        pts_array = np.array(points, dtype=np.int32)
        with open(ROI_FILE, "w") as f: json.dump(pts_array.tolist(), f)
        return pts_array
    return None


def show_productivity_page():
    if st.button("⬅️ BACK TO MENU"):
        st.session_state['page'] = 'main'
        st.rerun()

    st.title("PRODUCTIVITY MONITOR")
    col_video, col_stats = st.columns([3, 1])
    
    # Initialize a trigger in session state if it doesn't exist
    if "timer_active" not in st.session_state:
        st.session_state["timer_active"] = False

    with col_stats:
        st.markdown("### Instructions\n1. Click button below\n2. Define ROI in pop-up\n3. Press **ENTER** to start")
        
        # Action Button
        btn_start = st.button("SET ROI & START TIMER")
        
        # Checkbox only controls manual stop/start
        run_monitor = st.checkbox("START MONITORING", value=st.session_state["timer_active"], key="prod_check")
        
        st.divider()
        timer_text, score_metric, status_display = st.empty(), st.empty(), st.empty()
    
    with col_video:
        frame_window = st.empty()

    # Load existing ROI
    pts = None
    if os.path.exists(ROI_FILE):
        with open(ROI_FILE, "r") as f: 
            pts = np.array(json.load(f), dtype=np.int32)

    # If user clicks the big start button
    if btn_start:
        pts = get_roi_interactively(UDP_URL)
        if pts is not None:
            st.session_state["timer_active"] = True
            st.rerun() # Rerun to refresh the UI and start the loop

    # Use run_monitor (the checkbox) or the timer_active state to enter the loop
    if run_monitor and pts is not None:
        vs = VideoStream(UDP_URL).start()
        model = YOLO("yolov8s.pt")
        DURATION, GRACE = 120, 15
        
        start_s = time.time()
        prod_s, unprod_s, last_l, idle_s, phone_cd, prev_g = 0, 0, time.time(), None, 0, None

        try:
            while True:
                curr = time.time()
                elapsed = curr - start_s
                
                # Exit if time is up or user unchecks the box
                if elapsed > DURATION or not st.session_state.get("prod_check"):
                    st.session_state["timer_active"] = False
                    break

                ret, frame = vs.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                delta = curr - last_l
                last_l = curr

                # --- PROD LOGIC ---
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                roi_img = cv2.bitwise_and(frame, frame, mask=mask)
                
                gray = cv2.GaussianBlur(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY), (5,5), 0)
                motion = np.sum(cv2.threshold(cv2.absdiff(prev_g, gray), 25, 255, cv2.THRESH_BINARY)[1])/255 if prev_g is not None else 0
                prev_g = gray
                hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
                hand = np.sum(cv2.bitwise_not(cv2.inRange(hsv, np.array([8, 60, 60]), np.array([30, 255, 200]))))/255 > 2000

                res = model(frame, conf=0.3, verbose=False)[0]
                phone = False
                for box in res.boxes:
                    if model.names[int(box.cls[0])] == "cell phone":
                        phone = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                phone_cd = 10 if phone else max(0, phone_cd - 1)
                active = hand and motion > 500
                status, dot = "PRODUCTIVE", "#00FF88"

                if phone_cd > 0:
                    status, dot = "UNPRODUCTIVE (PHONE)", "#FF0000"; unprod_s += delta; idle_s = None
                elif active:
                    prod_s += delta; idle_s = None
                else:
                    if idle_s is None: idle_s = time.time()
                    dot = "#FF0000"
                    if (time.time() - idle_s) > GRACE:
                        status = "UNPRODUCTIVE (IDLE)"; unprod_s += delta
                    else: status = "PRODUCTIVE"

                # --- VISUALS ---
                total = prod_s + unprod_s
                score = int((prod_s / total) * 100) if total > 0 else 0
                timer_text.metric("Time Remaining", f"{max(0, int(DURATION - elapsed))}s")
                score_metric.metric("Score", f"{score}%")
                status_display.markdown(f"<div style='border-left:10px solid {dot}; background:#111; padding:20px;'><h2 style='color:{dot};'>{status}</h2></div>", unsafe_allow_html=True)
                
                cv2.polylines(frame, [pts], True, (255, 255, 0), 2)
                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
        finally:
            vs.stop()
            cv2.destroyAllWindows()

# --- 6. ROUTER ---

if st.session_state['page'] == 'main': 
    show_main_menu()
elif st.session_state['page'] == 'geofence': 
    show_geofence_page()
elif st.session_state['page'] == 'productivity': 
    show_productivity_page()
elif st.session_state['page'] == 'ppe': # Added this line
    show_ppe_page()