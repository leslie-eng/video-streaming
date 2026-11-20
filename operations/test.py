import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import requests
import streamlit as st
import urllib3
from dotenv import load_dotenv
from ultralytics import YOLO
from ultralytics.engine.results import Results


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OPERATIONS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = OPERATIONS_DIR.parent
load_dotenv(OPERATIONS_DIR / ".env")

stream_url = os.getenv("STREAM_URL")

log_dir = PROJECT_DIR / "logs" / "stream.log"
save_dir = PROJECT_DIR / "storage" / "screenshots"
rec_dir = PROJECT_DIR / "storage" / "recordings"

log_dir.parent.mkdir(parents=True, exist_ok=True)
save_dir.mkdir(parents=True, exist_ok=True)
rec_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_dir,
    format="%(asctime)s %(levelname)s: %(message)s",
    filemode="w",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class StreamInterruptedError(Exception):
    pass


@st.cache_resource
def load_model() -> YOLO:
    model_path = OPERATIONS_DIR / "yolo11n.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    return YOLO(str(model_path))


st.sidebar.header("Camera & Detection Controls")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
scale_factor = st.sidebar.slider("Frame Scale", 0.25, 1.0, 0.5, 0.05)
run_stream = st.sidebar.toggle("Start Camera Feed", value=False)

if "recording" not in st.session_state:
    st.session_state.recording = False
if "video_writer" not in st.session_state:
    st.session_state.video_writer = None
if "take_screenshot" not in st.session_state:
    st.session_state.take_screenshot = False

if st.sidebar.button("Capture Screenshot"):
    st.session_state.take_screenshot = True

rec_label = "Stop Recording" if st.session_state.recording else "Start Recording"
if st.sidebar.button(rec_label):
    st.session_state.recording = not st.session_state.recording
    if not st.session_state.recording:
        stop_writer = st.session_state.video_writer
        if stop_writer is not None:
            stop_writer.release()
            st.session_state.video_writer = None
        st.sidebar.success("Recording saved")

status_placeholder = st.empty()
frame_placeholder = st.empty()


def stop_recording() -> None:
    writer = st.session_state.video_writer
    if writer is not None:
        writer.release()
        st.session_state.video_writer = None
    st.session_state.recording = False


def report_error(message: str) -> None:
    status_placeholder.error(message)
    logger.error(message)


try:
    model = load_model()
except Exception as error:
    model = None
    report_error(f"Unable to load YOLO model: {error}")


def stream_camera(url: str, timeout: float = 30.0, max_attempts: int = 3) -> None:
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        response = None
        frame_count = 0
        bytes_data = b""
        prev_time = time.perf_counter()
        fps = 0.0
        last_frame_time = time.perf_counter()

        try:
            response = requests.get(url, verify=False, stream=True, timeout=timeout)
            response.raise_for_status()

            if response.encoding:
                content_type = response.headers.get("Content-Type", "")
                if "multipart" not in content_type.lower() and "image/jpeg" not in content_type.lower():
                    report_error(
                        f"Unexpected stream content type: {content_type or 'not provided'}. "
                        "Expected an MJPEG endpoint."
                    )
                    logger.error("Unexpected stream content type: %s", content_type)
                    return

            for chunk in response.iter_content(chunk_size=16384):
                if not run_stream:
                    return

                if time.perf_counter() - last_frame_time > timeout:
                    raise StreamInterruptedError(
                        f"No complete frames were received for {timeout:.0f} seconds"
                    )

                if not chunk:
                    continue

                bytes_data += chunk

                while True:
                    start = bytes_data.find(b"\xff\xd8")
                    end = bytes_data.find(b"\xff\xd9")

                    if start == -1:
                        bytes_data = b""
                        break
                    if end == -1:
                        if start > 0:
                            bytes_data = bytes_data[start:]
                        break
                    if end < start:
                        bytes_data = bytes_data[start:]
                        continue

                    jpg = bytes_data[start : end + 2]
                    bytes_data = bytes_data[end + 2 :]

                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    last_frame_time = time.perf_counter()
                    frame_count += 1

                    curr_time = time.perf_counter()
                    time_diff = curr_time - prev_time
                    if time_diff > 0:
                        instant_fps = 1.0 / time_diff
                        fps = instant_fps if fps == 0.0 else (0.9 * fps) + (0.1 * instant_fps)
                    prev_time = curr_time

                    resized_frame = cv2.resize(
                        frame,
                        (0, 0),
                        fx=scale_factor,
                        fy=scale_factor,
                    )
                    height, width = resized_frame.shape[:2]

                    if model is None:
                        report_error("The YOLO model is unavailable")
                        return

                    results = model(resized_frame, conf=conf_threshold, verbose=False)
                    result = cast(Results, next(iter(results)))
                    annotated_frame = result.plot()

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(
                        annotated_frame,
                        f"{timestamp} | {width}x{height} | FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    if st.session_state.take_screenshot:
                        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        filepath = save_dir / filename
                        if cv2.imwrite(str(filepath), annotated_frame):
                            status_placeholder.toast(f"Saved screenshot: {filename}")
                        else:
                            report_error(f"Could not save screenshot: {filepath}")
                        st.session_state.take_screenshot = False

                    if st.session_state.recording:
                        if st.session_state.video_writer is None:
                            rec_fps = max(1.0, round(fps)) if fps > 0 else 15.0
                            fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
                            filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                            filepath = rec_dir / filename
                            writer = cv2.VideoWriter(
                                str(filepath),
                                fourcc,
                                rec_fps,
                                (width, height),
                            )
                            if not writer.isOpened():
                                writer.release()
                                st.session_state.recording = False
                                report_error(f"Could not start recording: {filepath}")
                            else:
                                st.session_state.video_writer = writer

                        cv2.circle(annotated_frame, (width - 25, 25), 8, (0, 0, 255), -1)
                        cv2.putText(
                            annotated_frame,
                            "REC",
                            (width - 75, 31),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        writer = st.session_state.video_writer
                        if writer is not None:
                            writer.write(annotated_frame)

                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        rgb_frame,
                        channels="RGB",
                        use_container_width=True,
                    )

                    break

            if frame_count == 0:
                raise StreamInterruptedError("The stream ended without delivering any usable JPEG frames")
            raise StreamInterruptedError("The camera stream ended unexpectedly")

        except requests.exceptions.HTTPError as error:
            status_code = error.response.status_code if error.response is not None else "unknown"
            last_error = f"The camera URL returned HTTP {status_code}"
            report_error(last_error)
            logger.error("HTTP error %s for %s", status_code, url)
            return

        except requests.exceptions.Timeout:
            last_error = f"Connection timed out after {timeout} seconds"

        except requests.exceptions.ConnectionError as error:
            last_error = f"Could not connect to the camera: {error}"

        except requests.exceptions.RequestException as error:
            last_error = f"Camera stream request failed: {error}"

        except StreamInterruptedError as error:
            last_error = str(error)

        except Exception as error:
            last_error = f"Unexpected streaming error: {error}"

        finally:
            if response is not None:
                response.close()

        if attempt < max_attempts:
            message = f"{last_error}. Retrying {attempt + 1}/{max_attempts}..."
            report_error(message)
            time.sleep(2)

    report_error(f"{last_error}. Retrying stopped after {max_attempts} attempts.")


if not stream_url:
    report_error("STREAM_URL is not set. Add it to operations/.env.")
elif model is None:
    pass
elif run_stream:
    try:
        stream_camera(stream_url)
    finally:
        stop_recording()
else:
    stop_recording()
    status_placeholder.info("Enable **Start Camera Feed** in the sidebar to begin streaming.")
