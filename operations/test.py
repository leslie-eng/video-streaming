import cv2
import numpy as np
import requests
import urllib3
from datetime import datetime
import time
from dotenv import load_dotenv
import os
from pathlib import Path
from ultralytics import YOLO
from typing import cast
from ultralytics.engine.results import Results
import streamlit as st

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

save_dir = Path("storage/screenshots")
save_dir.mkdir(parents=True, exist_ok=True)

rec_dir = Path("storage/recordings")
rec_dir.mkdir(parents=True, exist_ok=True)

# Initialize session state for recording & screenshots
if "recording" not in st.session_state:
    st.session_state.recording = False
if "video_writer" not in st.session_state:
    st.session_state.video_writer = None
if "take_screenshot" not in st.session_state:
    st.session_state.take_screenshot = False

# Sidebar controls
st.sidebar.header("Camera & Detection Controls")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
scale_factor = st.sidebar.slider("Frame Scale", 0.25, 1.0, 0.5, 0.05)

run_stream = st.sidebar.toggle("Start Camera Feed", value=False)

if st.sidebar.button("Capture Screenshot"):
    st.session_state.take_screenshot = True

rec_label = "Stop Recording" if st.session_state.recording else "Start Recording"
if st.sidebar.button(rec_label):
    st.session_state.recording = not st.session_state.recording
    if not st.session_state.recording and st.session_state.video_writer is not None:
        st.session_state.video_writer.release()
        st.session_state.video_writer = None
        st.sidebar.success("Recording saved!")

# Placeholders for live video and status messages
status_placeholder = st.empty()
frame_placeholder = st.empty()

if run_stream:
    try:
        stream_url = os.getenv("STREAM_URL")
        stream = requests.get(stream_url, verify=False, stream=True, timeout=5)
        bytes_data = b""
        prev_time = time.perf_counter()
        fps = 0.0

        for chunk in stream.iter_content(chunk_size=16384):
            if not run_stream:
                break

            bytes_data += chunk
            start = bytes_data.find(b"\xff\xd8")
            end = bytes_data.find(b"\xff\xd9")

            if start != -1 and end != -1 and end < start:
                bytes_data = bytes_data[start:]
                continue

            if start != -1 and end != -1 and end > start:
                jpg = bytes_data[start : end + 2]
                bytes_data = bytes_data[end + 2 :]

                if len(jpg) > 0:
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if frame is not None:
                        # Calculate smoothed FPS
                        curr_time = time.perf_counter()
                        time_diff = curr_time - prev_time
                        if time_diff > 0:
                            instant_fps = 1.0 / time_diff
                            fps = instant_fps if fps == 0.0 else (0.9 * fps) + (0.1 * instant_fps)
                        prev_time = curr_time

                        # Resize frame
                        resized_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                        h, w, _ = resized_frame.shape

                        # Run YOLO detection (note results[0].plot())
                        results = model(resized_frame, conf=conf_threshold, verbose=False)
                        result = cast(Results, next(iter(results)))
                        annotated_frame = result.plot()

                        # Draw timestamp, resolution, and FPS overlay
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        overlay_text = f"{timestamp} | {w}x{h} | FPS: {fps:.1f}"
                        cv2.putText(
                            annotated_frame,
                            overlay_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                        # Handle Screenshot button
                        if st.session_state.take_screenshot:
                            filepath = save_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(str(filepath), annotated_frame)
                            status_placeholder.toast(f"Saved: {filepath.name}")
                            st.session_state.take_screenshot = False

                        # Handle Recording toggle
                        if st.session_state.recording:
                            if st.session_state.video_writer is None:
                                rec_fps = max(1.0, round(fps)) if fps > 0 else 15.0
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                                rec_path = rec_dir / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                                st.session_state.video_writer = cv2.VideoWriter(
                                    str(rec_path), fourcc, rec_fps, (w, h)
                                )
                            st.session_state.video_writer.write(annotated_frame)
                            cv2.circle(annotated_frame, (w - 25, 25), 8, (0, 0, 255), -1)
                            cv2.putText(
                                annotated_frame,
                                "REC",
                                (w - 75, 31),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2,
                                cv2.LINE_AA,
                            )

                        # Convert BGR to RGB and render in Streamlit
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    except Exception as e:
        st.error(f"Stream connection error: {e}")
else:
    if st.session_state.video_writer is not None:
        st.session_state.video_writer.release()
        st.session_state.video_writer = None
        st.session_state.recording = False
    st.info("Toggle **Start Camera Feed** in the sidebar to begin streaming.")

# stream_url = "https://192.168.100.7:8080/video"

# # # try:
# # #     response = requests.get(stream_url, verify=False, stream=True, timeout=5)
# # #     print("Status code:", response.status_code)
    
# # # except Exception as e:
# # #     print(f"the error was {e}")

# cap = cv2.VideoCapture(stream_url)

# if not cap.isOpened():
#     print("Error: Could not open video stream.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to grab frame.")
#         break

#     cv2.imshow("Live Camera Feed", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()