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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

model = YOLO("yolo11n.pt")

save_dir = Path("storage/screenshots")
save_dir.mkdir(parents=True, exist_ok=True)

rec_dir = Path("storage/recordings")
rec_dir.mkdir(parents=True, exist_ok=True)

recording = False
video_writer = None

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

url = os.getenv("STREAM_URL")
if url is None:
    raise ValueError("STREAM_URL environment variable is not set")

stream = requests.get(url, verify=False, stream=True, timeout=20)

prev_time = time.perf_counter()
fps = 0.0



bytes_data = b""
for chunk in stream.iter_content(chunk_size=4096):
    bytes_data += chunk
    start = bytes_data.find(b"\xff\xd8")  # Start of JPEG frame
    end = bytes_data.find(b"\xff\xd9")    # End of JPEG frame

    # Discard any incomplete trailing data before the first JPEG start marker
    if start != -1 and end != -1 and end < start:
        bytes_data = bytes_data[start:]
        continue

    if start != -1 and end != -1 and end > start:
        jpg = bytes_data[start : end + 2]
        bytes_data = bytes_data[end + 2 :]

        if len(jpg) > 0:
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is not None:
                curr_time = time.perf_counter()
                time_diff = curr_time - prev_time
                
                if time_diff > 0:
                    instant_fps = 1.0 /time_diff
                    fps = instant_fps if fps == 0.0 else (0.9 * fps) + (0.1 * instant_fps)
                prev_time = curr_time
                
                resized_frame = cv2.resize(frame,(0,0), fx=0.5, fy=0.5)
                
                # Run YOLO object detection and draw boxes
                results = model(resized_frame, conf=0.25, verbose=False)
                result = cast(Results, next(iter(results)))
                annotated_frame = result.plot()
                
                og_height, og_width, _ = annotated_frame.shape

                # Live timestamp instead of the one frozen at startup
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                info_text = f"{now_str} | IP Camera | {og_width}x{og_height} | FPS: {fps:.1f}"
                cv2.putText(annotated_frame, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                if recording and video_writer is not None:
                    # Draw the REC indicator first so it ends up in the saved video too
                    cv2.circle(annotated_frame, (og_width - 25, 25), 8, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, "REC", (og_width - 75, 31),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    video_writer.write(annotated_frame)

                cv2.imshow("Live Camera Feed", annotated_frame)

                # Press 'q' to quit
                key = cv2.waitKey(1) & 0XFF
                
                if key == ord("r"):
                    if not recording:
                        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        rec_fps = max(1.0, round(fps)) if fps > 0 else 20.0
                        fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
                        rec_path = rec_dir / f"video_{ts}.mp4"
                        video_writer = cv2.VideoWriter(str(rec_path), fourcc, rec_fps, (og_width, og_height))
                        recording = True
                        print(f"Started recording: {rec_path.resolve()}")
                    else:
                        recording = False
                        if video_writer is not None:
                            video_writer.release()
                        video_writer = None 
                        print("Stopped recording.")
                        
                
                if key == ord("s"):
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"screenshot_{ts}.jpg"
                    filepath = save_dir / filename
                    cv2.imwrite(str(filepath), annotated_frame)
                    print(f"Saved: {filepath.resolve()}")
                    
                elif key == ord("q"):
                    break

cv2.destroyAllWindows()

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