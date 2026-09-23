# Real-time dashcam object detection (RT-DETR + Flask)

This is a personal project where I try to do real-time object detection on a live dashcam feed with [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd), running on an NVIDIA GPU. The camera is a typical consumer dashcam, an "ULTRA HD DV 4K" camera (16.0 MP, 4K at 30 fps, Wi-Fi). Connected over USB in PC-camera mode, it shows up as a standard UVC webcam, and the app streams the annotated video with live stats to a browser on the local network.

Live camera → RT-DETRv2 (PyTorch, CUDA) → boxes, labels and scores → MJPEG stream plus live stats in the browser.

```
capture thread ──(latest-frame slot)──▶ inference thread ──(latest-result slot)──▶ Flask (MJPEG + JSON API) ──▶ browser
 OpenCV V4L2/FFmpeg                      GPU preprocess → RT-DETR fp16 → draw → JPEG (once)
```

The buffers are single-slot "latest value wins" slots, not queues. If inference falls behind, frames are dropped and latency stays bounded. The design follows an internal implementation plan; section references like "PLAN §12" in code comments refer to it.

## Development setup

| | |
|---|---|
| GPU | NVIDIA RTX 5060 Ti 16 GB (Blackwell, sm_120), driver 580 |
| Python env | `.venv` (Python 3.12), torch 2.14.0+cu130, transformers 5.17.0, opencv-python-headless 5.0 |
| Camera | `/dev/video0`: generic USB UVC camera, MJPG 1280x720 (`/dev/video1` is a metadata node) |
| Model | `PekingU/rtdetr_v2_r50vd` (COCO, 53.4 AP), fp16 autocast |
| UI | `http://<server-ip>:8000` (bound to `0.0.0.0`; see [Network exposure](#network-exposure)) |

## Setup

System packages (already installed here):

```bash
sudo apt install -y v4l-utils ffmpeg python3-venv python3-dev
sudo usermod -aG video "$USER"      # camera access; then log out and back in
```

Python environment:

```bash
cd ~/projects/real-time-detection
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
# RTX 50-series (Blackwell) needs a CUDA >= 12.8 build of torch; this env uses CUDA 13.0 wheels:
.venv/bin/pip install -r requirements.txt       # includes --extra-index-url .../whl/cu130
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- Install only `opencv-python-headless`. Having `opencv-python` installed as well breaks `cv2`.
- The model weights download from Hugging Face on first run and are cached in `~/.cache/huggingface`. After that, you can run with `HF_HUB_OFFLINE=1` to skip the Hub checks (for example in the car). This also removes the "unauthenticated requests" warning.

## Running

```bash
cd ~/projects/real-time-detection
.venv/bin/python -m app.main                                  # uses config.yaml (live camera /dev/video0)
.venv/bin/python -m app.main --source samples/clip.mp4        # recorded dashcam clip instead of the camera
.venv/bin/python -m app.main --host 127.0.0.1 --port 8080 --threshold 0.4 --checkpoint PekingU/rtdetr_v2_r18vd
```

Then open `http://127.0.0.1:8000` on the server, or `http://<server-ip>:8000` from another LAN device. The LAN URL only works once the firewall allows it; see [Firewall](#firewall-lan-access). Stop the app with Ctrl+C, which exits within about 1 s and releases the camera immediately.

To run it in the background:

```bash
setsid nohup .venv/bin/python -m app.main > logs/app.log 2>&1 < /dev/null &
pkill -INT -f "app.main"          # clean stop
```

### UI
- **Left:** the annotated live stream. It reconnects automatically with backoff.
- **Right:**
  - Stats, refreshed every 500 ms: capture FPS, inference FPS, inference ms, end-to-end latency, device and model.
  - Detection list with per-class counts, refreshed every 200 ms.
  - Score-threshold slider.
  - Class checkboxes with **Driving preset / All / None**.
- Setting changes take effect on the next frame. The UI has no external dependencies, so it works offline.

### HTTP API

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Browser UI |
| `/video_feed` | GET | MJPEG stream (`multipart/x-mixed-replace; boundary=frame`) |
| `/api/stats` | GET | capture/inference FPS, inference ms and e2e latency ms (mean/p95), frame id, detection count, device, model, resolution, uptime, per-stage timings |
| `/api/detections` | GET | latest frame id + `[{x1,y1,x2,y2,score,class_id,label}]` |
| `/api/labels` | GET | all class names, `id2label`, and the driving preset (in the checkpoint's own spelling) |
| `/api/settings` | GET/POST | `{"score_threshold": 0.4, "classes": ["person","car"]}`. Either field is optional. Invalid input gets a 400 with `{"error": ...}` and nothing is applied. |
| `/healthz` | GET | 200 if capture and inference both produced output within the last 5 s, otherwise 503 |

```bash
curl -s localhost:8000/api/stats
curl -s -H 'Content-Type: application/json' -d '{"score_threshold":0.4}' localhost:8000/api/settings
```

## Configuration (`config.yaml`)

Every key is optional; the defaults live in `app/config.py`. Unknown keys and invalid values are rejected at startup with a clear message. CLI overrides: `--config --source --checkpoint --device --host --port --threshold --log-level`.

| Section | Key | Default | Notes |
|---|---|---|---|
| camera | `source` | `/dev/video0` | index, `/dev/videoN`, video file, `rtsp://…`, `http://…` |
| | `backend` | `auto` | V4L2 for devices, FFmpeg for files/URLs |
| | `width`/`height` | 1280/720 | the model always sees 640x640, so lower capture sizes barely help (see benchmark) |
| | `fps` | 25 | see [camera frame rate](#camera-frame-rate-25-fps) |
| | `fourcc` | `MJPG` | needed for 720p over USB; `""` = don't set |
| | `loop_file`, `pace_file` | true | loop files and play them at native FPS (simulates a camera) |
| | `reconnect_delay_s`, `max_consecutive_failures` | 2.0, 30 | reconnect after the camera is unplugged |
| | `first_frame_timeout_s` | 10 | startup fails with hints if no frame arrives in this time |
| model | `checkpoint` | `PekingU/rtdetr_v2_r50vd` | r18vd / r34vd / r50vd / r101vd |
| | `device` | `auto` | `cuda`, `cuda:0`, `cpu` (CPU gives only a few FPS) |
| | `fp16` | true | autocast; post-processing runs in fp32 |
| | `score_threshold` | 0.5 | adjustable live in the UI |
| | `classes` | driving subset | `[]` = all 80. COCO spellings are accepted (e.g. `motorcycle` maps to the checkpoint's `motorbike`) |
| | `compile` | false | experimental `torch.compile` (see benchmark) |
| | `warmup_iters` | 10 | runs inside the inference thread before serving |
| render | `line_thickness`, `font_scale`, `draw_hud`, `output_width` | 2, 0.5, true, 0 | `output_width: 960` shrinks stream bandwidth |
| server | `host` | `0.0.0.0` | see [Network exposure](#network-exposure) |
| | `port`, `jpeg_quality`, `max_stream_fps` | 8000, 80, 30 | |
| logging | `level` | INFO | a stats line is logged every 5 s |

**Label spellings:** the PekingU checkpoints use VOC-style names in `id2label`: `motorbike`, `sofa`, `aeroplane`, `tvmonitor`, `pottedplant`, `diningtable`. Both spellings are accepted in the config and the API, but the UI and API report the checkpoint's own spelling. The full label list is logged at startup, and unknown names in `config.yaml` trigger a warning and are ignored.

## Model choice and performance

Benchmark: `scripts/benchmark.py`, run on an idle RTX 5060 Ti with 300 frames of `samples/clip.mp4` at 1280x720, fp16 autocast, eager mode. "Total" is preprocess + forward + post-process + drawing + JPEG encode (encode ≈ 1.35 ms, drawing ≈ 0.2 ms). Times are in ms, mean/p95.

| Checkpoint | COCO AP | Forward | Total | Max FPS | GPU mem |
|---|---|---|---|---|---|
| r18vd | 48.1 | 7.7 / 7.8 | 10.1 / 10.2 | 99 | 262 MB |
| r34vd | 49.9 | 10.0 / 10.1 | 12.3 / 12.4 | 81 | 320 MB |
| **r50vd (default)** | **53.4** | 16.5 / 16.6 | **18.8 / 19.0** | 53 | 386 MB |
| r101vd | 54.3 | 23.3 / 23.4 | 25.6 / 25.7 | 39 | 518 MB |

AP values are from the official RT-DETRv2 README.

- **Other configurations:**
  - `compile: true` (reduce-overhead): r18 6.9, r50 12.4 / 13.6, r101 15.1 / 15.5 ms total. Output matches eager (IoU ≥ 0.997), but warmup takes 6–14 s, so it stays experimental and off.
  - fp32 (`fp16: false`): slower. r50 totals 24.3 / 30.9 ms.
  - Lower capture resolution: r50 at 800x600 is 18.2 ms and at 640x480 is 18.0 ms. That's almost no gain, so 720p stays.
- **Why r50vd:** it is +5.5 AP over r18, yet sustains more than 50 FPS at p95, about 2x headroom over the 25 fps camera. r101 adds only +0.9 AP with much less headroom.

**Measured end to end.** This was the full app on the live camera with r50vd and one browser stream client connected, over 150 s (30 stats windows):

| Metric | Result | Target (PLAN §12) |
|---|---|---|
| Capture FPS | 25.1 | camera limit |
| Inference FPS | 25.1–25.2 (no dropped frames, 0 errors) | ≥ camera FPS |
| Inference ms | 17.7 mean, worst window p95 18.3 | |
| Server e2e latency (capture → JPEG published) | 18.8 mean, worst window p95 19.5 | p95 < 100 ms |
| GPU memory (process) | ≈ 560 MB | |

On the sample clip with r18, the headless run gave 30 fps in and out, inference about 10 ms, and e2e p95 27.6 ms over 130 s. With an artificially slow detector (200 ms/frame), e2e latency stayed flat at 219 ms mean / 234 ms p95, confirming that stale frames are dropped instead of queued.

To reproduce:

```bash
.venv/bin/python scripts/benchmark.py --source samples/clip.mp4 --frames 300 --checkpoints PekingU/rtdetr_v2_r18vd,PekingU/rtdetr_v2_r50vd
.venv/bin/python scripts/headless_run.py --source samples/clip.mp4 --seconds 300          # capture + inference, no Flask
.venv/bin/python scripts/headless_run.py --source samples/clip.mp4 --seconds 30 --fake-detector --fake-delay 0.2
```

## Camera frame rate: 25 fps

This camera advertises only 30 fps modes, but it actually delivers **25 fps** at 1280x720, 800x600 and 640x480 alike. That was measured with `ffmpeg -f v4l2` as well as with our probe, at a steady 40 ms interval. The camera exposes no V4L2 controls to change this. `camera.fps` is therefore set to 25. The driver still negotiates "30", so startup logs a harmless `requested 25 fps but driver negotiated 30.00 fps` warning. The pipeline itself could handle 50+ FPS with r50.

## Phase 0: camera discovery commands

```bash
lsusb                                          # is the camera enumerated?
v4l2-ctl --list-devices                        # which /dev/videoN nodes belong to it
v4l2-ctl -d /dev/video0 --list-formats-ext     # formats / resolutions / fps
v4l2-ctl -d /dev/video0 --all
ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -framerate 30 -i /dev/video0 -t 8 -f null -   # real delivered fps
groups | grep -q video || sudo usermod -aG video "$USER"                                            # then re-login

.venv/bin/python scripts/probe_camera.py --source /dev/video0 --seconds 10 --save-frames 5          # negotiated props, measured FPS, frames
.venv/bin/python scripts/probe_camera.py --source /dev/video0 --seconds 30 --record samples/live_clip.mp4
.venv/bin/python scripts/test_image.py --image samples/coco/000000039769.jpg --out samples/cats_det.jpg [--reference]
```

- A dashcam that only shows up as a USB drive is in mass-storage mode; switch it to "PC camera / UVC" mode in its menu.
- A Wi-Fi dashcam needs its RTSP URL: `--source rtsp://…`, which uses TCP transport automatically.
- An HDMI output needs a USB capture card, which then appears as `/dev/videoN`.

## Network exposure

`server.host: 0.0.0.0` makes the **unauthenticated** camera feed and settings API reachable by every device that can reach port 8000. That is fine on a trusted home LAN, but don't forward the port to the internet. Use `--host 127.0.0.1`, or set it in `config.yaml`, for local-only access.

### Firewall (LAN access)

If `ufw` is active, it blocks port 8000 from other machines by default. To view the UI from another device on the LAN (replace `<lan-subnet>` with yours, e.g. `192.168.1.0/24`):

```bash
sudo ufw allow from <lan-subnet> to any port 8000 proto tcp
```

Alternatively, without changing the firewall, use an SSH tunnel from your PC and open `http://localhost:8000`:

```bash
ssh -N -L 8000:localhost:8000 <user>@<server-ip>
```

## Tests

```bash
.venv/bin/python -m pytest -q                         # 113 tests, about 30 s (GPU tests auto-skip without CUDA/weights)
RTDETR_TEST_CHECKPOINT=PekingU/rtdetr_v2_r50vd .venv/bin/python -m pytest tests/test_detector.py
```

| File | Covers |
|---|---|
| `test_buffers.py` | overwrite semantics, `wait_newer` timeout, concurrent readers, close |
| `test_render.py` | empty list, boxes partly outside or at edges, tiny/degenerate boxes, long labels, HUD, output_width |
| `test_capture.py` | file source: pacing, looping, reopen, friendly errors (missing device, permission denied) |
| `test_pipeline.py` | worker with a fake detector, error resilience, **slow-detector latency bound** |
| `test_server.py` | all routes via the Flask test client, settings validation (400s), MJPEG framing |
| `test_detector.py` | two-cats image, Detection field validity, class filter and aliases, threshold, **GPU-vs-HF-processor parity** on 9 COCO images |
| `test_integration.py` | full app subprocess on the clip: stats, `/video_feed` boundary + JPEG SOI, settings 400/200, `/healthz`, SIGINT → exit 0 |

**Parity note:** the fast GPU preprocessing matches the HF processor to within 1/255 per pixel. At the detection level, r18 and r34 meet the strict criterion on every match (same label, IoU > 0.9, |Δscore| < 0.05). On r50 and r101, 1–3 of about 60 borderline, overlapping objects exceed the score bound. The HF reference path alone varies by a similar amount between runs, because cuDNN picks different algorithms each time. So the test requires the strict criterion for every image's top detection and for at least 90% of matches, and requires that every reference object is still found.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Permission denied opening /dev/video0` | `sudo usermod -aG video $USER`, then log out and back in (check with `groups`) |
| "device exists … probably busy" | Another app holds the camera. Find it with `fuser -v /dev/video0` and close it (only one process can open it) |
| No frames / wrong node | Use the node that lists formats in `v4l2-ctl --list-formats-ext`; `/dev/video1` here is metadata-only |
| Low FPS with YUYV | Keep `fourcc: MJPG` |
| FPS below the configured value | The camera delivers fewer frames than it advertises (here 25 vs 30); check with the ffmpeg command above |
| `CUDA is not available … running on CPU` | Check `nvidia-smi` and `torch.cuda.is_available()`. Blackwell GPUs need a cu128+ torch build |
| Slow first start / Hub warnings | The first run downloads weights; afterwards use `HF_HUB_OFFLINE=1` |
| UI unreachable from another device | Check `server.host` (0.0.0.0) and the firewall (`sudo ufw allow …`), or use an SSH tunnel |
| Stream shows "reconnecting" | The server was restarted or stopped. The page retries automatically with backoff |
| Import errors for `cv2` | `pip uninstall opencv-python`; keep only `opencv-python-headless` |
| Big "car" box over the bottom of the frame | The model sometimes detects your own car's hood. Mount the camera so less hood is visible, or raise the threshold |

## Project layout

```
app/        main.py (entrypoint/lifecycle), config.py, capture.py, detector.py, pipeline.py,
            render.py, server.py, buffers.py, stats.py
templates/  index.html          static/  app.js, style.css
scripts/    probe_camera.py, test_image.py, benchmark.py, headless_run.py
tests/      unit + integration tests (see above)
samples/    clip.mp4 (CC0 Wikimedia dashcam clip), live_clip.mp4, coco/ test images, SOURCES.txt
config.yaml, requirements.txt, LICENSE
```

## License

[MIT](LICENSE) © 2026 Petar I. Penchev
