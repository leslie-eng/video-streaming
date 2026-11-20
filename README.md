# Real-Time IP Camera Streaming

A Python application that reads an MJPEG stream from an IP camera, performs YOLO11 object detection with OpenCV, and displays the annotated video feed. The live view can record MP4 clips and save screenshots.

## Features

- Streams an MJPEG endpoint over HTTP/HTTPS
- Decodes JPEG frames and resizes them to 50% for processing
- Runs YOLO11 object detection on each frame
- Displays timestamps, camera label, resolution, and smoothed FPS
- Starts and stops MP4 recording with the `r` key
- Saves annotated screenshots with the `s` key
- Exits the application with the `q` key

## Technology Stack

- Python 3.14
- OpenCV for frame decoding, rendering, and video recording
- Ultralytics YOLO11 for object detection
- Requests for the HTTP stream connection
- python-dotenv for environment configuration
- NumPy for frame buffers
- uv for environment and dependency management

## Requirements

- Python 3.14 or later
- A camera that exposes an MJPEG or multipart JPEG video endpoint
- Network access to the camera
- A graphical desktop capable of opening an OpenCV window
- A compatible `yolo11n.pt` model file in `operations/`

## Installation

From the repository root:

```bash
uv sync
```

This creates the project environment and installs the dependencies declared in `pyproject.toml`.

To use the OpenCV window and YOLO model, run the application inside a graphical desktop session rather than through a headless server or container.

## Configuration

The application reads `STREAM_URL` from a `.env` file in `operations/`.

```dotenv
STREAM_URL=https://camera-address:8080/video
```

The URL must point to a live MJPEG stream. The current project `.env` is ignored by Git and can be kept local. Replace the example address with the camera endpoint used on your network.

> [!WARNING]
> The stream request disables TLS certificate verification (`verify=False`) because some IP cameras use self-signed certificates. Use this only on a trusted network.

## Running the Application

Run the executable script from the `operations` directory:

```bash
cd operations
uv run python test.py
```

The stream window opens as **Live Camera Feed**. Processing starts only after a complete JPEG frame is received.

## Controls

| Key | Action |
| --- | --- |
| `r` | Start or stop recording |
| `s` | Save the current annotated frame as a JPEG screenshot |
| `q` | Quit and close the window |

## Output Files

Files are stored relative to `operations/` because the script is run from that directory:

- Screenshots: `operations/storage/screenshots/screenshot_YYYY-MM-DD_HH-MM-SS.jpg`
- Recordings: `operations/storage/recordings/video_YYYY-MM-DD_HH-MM-SS.mp4`

The recording codec is MP4V and the writer uses a frame size matching the annotated, resized frame.

## How It Works

1. `test.py` loads `STREAM_URL` from `operations/.env`.
2. Requests opens the HTTP stream with streaming enabled and a 20-second timeout.
3. Incoming bytes are buffered until a complete JPEG frame marker (`FFD8` to `FFD9`) is found.
4. OpenCV decodes the JPEG and reduces it to 50% size.
5. Ultralytics runs YOLO11 inference with a confidence threshold of `0.25`.
6. The detected objects and stream metadata are drawn onto the frame.
7. The frame is displayed in an OpenCV window and optionally written to a recording or screenshot file.

## Project Structure

```text
video-stream/
├── operations/
│   ├── test.py              # Functional MJPEG streaming, detection, and capture entry point
│   ├── main.py              # Initial connection prototype
│   ├── yolo11n.pt           # YOLO11 nano model
│   ├── .env                 # Local STREAM_URL configuration
│   └── storage/
│       ├── screenshots/     # Saved JPEG screenshots
│       └── recordings/      # Saved MP4 recordings
├── src/video_stream/
│   └── __init__.py          # Placeholder package entry point
├── pyproject.toml           # Python and uv project configuration
├── uv.lock                  # Locked dependency versions
└── requirements.txt         # Reserved; currently no direct entries
```

> [!NOTE]
> `main.py` is an incomplete connection prototype. The working application entry point is `operations/test.py`.

## Troubleshooting

- **`STREAM_URL environment variable is not set`**: Confirm `operations/.env` exists and contains `STREAM_URL=...`.
- **The stream returns no images**: Verify the URL in a browser or camera app and confirm it serves MJPEG content.
- **A TLS or certificate error occurs**: The script already suppresses certificate verification; confirm the address and network route.
- **The YOLO model cannot load**: Run from `operations/` so `yolo11n.pt` is found, or update `test.py` to use an absolute model path.
- **No window appears**: Ensure you are running in a graphical desktop session and not an SSH or headless environment.
- **The recording cannot open**: Confirm the MP4V codec is available in your OpenCV build and that the output directory is writable.

## Testing

No automated test suite is currently included. The available `operations/test.py` script is the manual integration test and live-stream entry point.
