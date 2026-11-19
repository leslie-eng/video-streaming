import cv2
import numpy as np
import requests
import urllib3
from datetime import datetime
import time
from dotenv import load_dotenv
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

url = os.getenv("STREAM_URL")
if url is None:
    raise ValueError("STREAM_URL environment variable is not set")

stream = requests.get(url, verify=False, stream=True, timeout=5)

prev_time = time.perf_counter()
fps = 0.0

key = cv2.waitKey(1) & 0XFF

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
                
                og_height, og_width, _ = resized_frame.shape
                
                # cv2.putText(resized_frame, timestamp, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                info_text = f"{timestamp} | IP Camera |{og_width}x{og_height} | FPS: {fps:.1f}"
                
                cv2.putText(resized_frame, info_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow("Live Camera Feed", resized_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# cv2.destroyAllWindows()

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