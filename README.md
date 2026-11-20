# Real-Time IP Camera Streaming

A Streamlit application that receives an MJPEG camera stream, runs Ultralytics YOLO11 object detection, and displays the annotated feed in a browser. It supports screenshots, MP4 recordings, and resilient handling of unavailable or interrupted camera streams.

## Features

- Browser-based live MJPEG camera viewer
- Configurable YOLO confidence threshold
- Configurable frame scale
- Smoothed live FPS, timestamp, and processed resolution overlays
- Screenshot capture
- MP4 recording start and stop controls
- Validation of missing configuration, HTTP responses, and recording output
- Automatic retry for connection and mid-stream interruptions
- Clean closure of streaming responses and recording writers
- Local logging and ignored generated artifacts

## Technology Stack

- Python 3.14
- Streamlit 1.67+
- OpenCV
- Ultralytics YOLO11
- Requests
- NumPy
- python-dotenv
- uv

## Requirements

- Python 3.14 or later
- A camera with an accessible MJPEG or multipart JPEG endpoint
- Network access to the camera
- A compatible `operations/yolo11n.pt` model file

## Installation

Install the dependencies from the repository root:

```bash
uv sync
```

This uses `pyproject.toml` and `uv.lock` to create or update the project environment.

## Configuration

Create `operations/.env` with the camera stream URL:

```dotenv
STREAM_URL=https://camera-address:8080/video
```

The variable name is case-sensitive and must be `STREAM_URL`. The file is ignored by Git and can contain a private network address.

> [!WARNING]
> The application disables TLS certificate verification because many IP cameras use self-signed certificates. It should only be used on trusted networks.

## Running the Application

Start Streamlit from the `operations` directory:

```bash
cd operations
uv run streamlit run test.py
```

Streamlit normally opens the application at `http://localhost:8501`. Open that address in a browser if it does not launch automatically.

## Controls

Use the Streamlit sidebar to control the application:

| Control | Action |
| --- | --- |
| Confidence Threshold | Sets the YOLO detection confidence threshold |
| Frame Scale | Resizes the frame before detection |
| Start Camera Feed | Starts or stops the MJPEG stream |
| Capture Screenshot | Saves the latest annotated frame |
| Start/Stop Recording | Starts or stops an MP4 recording |

## Failure Handling

The application separates configuration and connection failures from stream interruptions:

- **Missing `STREAM_URL`**: Shows an actionable configuration error and does not attempt a connection.
- **Wrong or unreachable URL**: Reports connection errors or HTTP status codes such as 404 and does not retry HTTP client or server errors.
- **Camera switched off**: Reports a connection failure and retries up to three times, waiting two seconds between attempts.
- **Mid-stream drop**: Detects an unexpected stream end or no complete JPEG frame for 30 seconds, closes the response, and retries up to three times.
- **Unexpected content type**: Reports an endpoint that does not appear to provide MJPEG data.
- **Recording failure**: Validates that OpenCV opened the `VideoWriter`; if not, it releases the writer and stops recording.
- **Clean shutdown**: Responses are closed and recording writers are released when the feed stops or a streaming session exits.

A camera stream that returns no usable JPEG frames is considered interrupted rather than a successful idle connection.

## Generated Files

Generated media is stored relative to the project root:

- Screenshots: `storage/screenshots/screenshot_YYYYMMDD_HHMMSS.jpg`
- Recordings: `storage/recordings/video_YYYYMMDD_HHMMSS.mp4`
- Application log: `logs/stream.log`

The directories are created automatically when the application starts. Screenshots, recordings, logs, and environment files are ignored by Git.

## How It Works

1. `test.py` loads `STREAM_URL` from `operations/.env`.
2. Streamlit caches the YOLO11 model in `operations/yolo11n.pt`.
3. Requests downloads the MJPEG stream with TLS verification disabled and a 30-second timeout.
4. The application extracts complete JPEG frames using JPEG start and end markers.
5. OpenCV decodes and scales the frames.
6. YOLO11 produces annotated detections and Streamlit renders the RGB frame.
7. Screenshots and recordings are saved to the project `storage` directories.
8. Stream and recording resources are closed on stop, failure, or session exit.

## Project Structure

```text
video-stream/
├── operations/
│   ├── test.py              # Streamlit camera application
│   ├── main.py              # Previous application/connection prototype
│   ├── yolo11n.pt           # YOLO11 nano model
│   └── .env                 # Local STREAM_URL configuration
├── storage/
│   ├── screenshots/         # Generated screenshots, ignored by Git
│   └── recordings/          # Generated MP4 files, ignored by Git
├── logs/                    # Generated logs, ignored by Git
├── src/video_stream/        # Python package placeholder
├── pyproject.toml           # Project dependencies and metadata
├── uv.lock                  # Locked dependency versions
└── README.md
```

## Troubleshooting

- **`STREAM_URL is not set`**: Create `operations/.env` and set `STREAM_URL=...`.
- **HTTP 404 or 400**: Verify the exact camera URL and endpoint path.
- **Connection refused or timed out**: Confirm the camera is powered on and reachable from the machine running Streamlit.
- **Stream ends unexpectedly**: Check camera sleep settings, Wi-Fi/network stability, and camera authentication.
- **Unexpected content type**: Confirm the endpoint returns MJPEG, usually `multipart/x-mixed-replace` or JPEG data.
- **The YOLO model cannot load**: Run from `operations/` or confirm `operations/yolo11n.pt` exists.
- **A screenshot or recording is missing**: Check `storage/screenshots/` or `storage/recordings/` under the project root.
- **Recording cannot start**: Confirm MP4V codec support and that the output directory is writable.
- **No Streamlit page opens**: Open `http://localhost:8501` manually after the server starts.

## Testing

There is currently no automated test suite. Manually verify the integration by testing each scenario:

1. Start with no `STREAM_URL` and confirm the configuration error.
2. Use an invalid URL and confirm HTTP or connection feedback.
3. Start with the camera off, then power it on and verify retry recovery.
4. Enable the stream and confirm YOLO, FPS, and timestamp overlays.
5. Capture one screenshot and confirm it appears in `storage/screenshots/`.
6. Record a short clip, stop recording, and confirm it appears in `storage/recordings/`.
7. Disconnect the camera during streaming and confirm retry, response closure, and recording cleanup.
8. Toggle off the stream and confirm the recording writer is released.