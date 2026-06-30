"""Live webcam cube-detection frontend: stream the detector's boxes to a browser (M4 check).

A tiny Flask server opens the webcam, runs the trained YOLO detector on each frame, draws the
boxes, and streams the annotated video to ``http://localhost:5000`` as MJPEG. Point your
webcam at a real cube to see whether the synthetic-trained detector transfers to reality --
the brief's true "done" bar for M4. Runs in the project venv (needs the ``demo`` extra: flask).

    python -m cube_tracker.eval.webcam_demo \
        --weights models_out/detector/yolo11n_cube_v2/weights/best.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator

import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Cube Detector — Live</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif;
           display: flex; flex-direction: column; align-items: center; }
    h1 { font-weight: 600; margin: 18px 0 4px; }
    p { color: #888; margin: 0 0 16px; }
    img { max-width: 96vw; border-radius: 10px; box-shadow: 0 6px 30px rgba(0,0,0,.6); }
  </style>
</head>
<body>
  <h1>🧊 Cube Detector — Live</h1>
  <p>Point your webcam at a Rubik's cube. Boxes are drawn by the synthetic-trained detector.</p>
  <img src="/stream" alt="webcam stream" />
</body>
</html>
"""


def _stream_frames(model: YOLO, camera: int, conf: float, imgsz: int) -> Iterator[bytes]:
    """Yield annotated webcam frames as an MJPEG multipart stream."""
    capture = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            encoded, buffer = cv2.imencode(".jpg", result.plot())
            if not encoded:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    finally:
        capture.release()


def create_app(weights: str, camera: int, conf: float, imgsz: int) -> Flask:
    app = Flask(__name__)
    model = YOLO(weights)

    @app.route("/")
    def index() -> str:
        return render_template_string(_PAGE)

    @app.route("/stream")
    def stream() -> Response:
        return Response(
            _stream_frames(model, camera, conf, imgsz),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam cube-detection demo.")
    parser.add_argument("--weights", required=True, help="Path to the trained detector .pt.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app = create_app(args.weights, args.camera, args.conf, args.imgsz)
    print(f"[webcam_demo] open http://{args.host}:{args.port} — Ctrl+C to stop")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
