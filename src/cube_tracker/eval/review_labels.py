"""Browser tool to verify Grounding DINO pseudo-labels before fine-tuning (M4 hardening).

Shows each real frame with its auto-label box (or a "NO CUBE" banner for hard negatives) one at
a time. Three actions per frame:

- **Keep** (K / Space / right): the label is correct -- a good cube box, or a genuinely empty
  negative -- leave it as is.
- **No cube** (N): there is no cube but a box was drawn (a false positive on a poster/face);
  clear the box so the frame becomes a clean hard negative instead of being thrown away.
- **Discard** (D): the frame is unusable; move it to ``_discarded/``.

With ``--recover`` the tool reviews a discard pile instead: Keep / No cube move the frame back
up to the parent folder (with its label kept or cleared); Discard leaves it discarded. Needs
the ``demo`` extra (flask).

    python -m cube_tracker.eval.review_labels --frames-dir data/real_frames
    python -m cube_tracker.eval.review_labels --frames-dir data/real_frames/_discarded --recover
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

_IMG_EXT = {".jpg", ".jpeg", ".png"}


def _frames(frames_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in frames_dir.iterdir()
        if p.suffix.lower() in _IMG_EXT and p.with_suffix(".txt").exists()
    )


def _overlay(path: Path) -> bytes:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    text = path.with_suffix(".txt").read_text(encoding="utf-8").strip()
    if text:
        cx, cy, bw, bh = (float(v) for v in text.split()[1:5])
        x1, y1 = int((cx - bw / 2) * width), int((cy - bh / 2) * height)
        x2, y2 = int((cx + bw / 2) * width), int((cy + bh / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 230, 0), 3)
    else:
        cv2.putText(
            image, "labeled: NO CUBE", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2
        )
    return bytes(cv2.imencode(".jpg", image)[1].tobytes())


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Label Review</title>
<style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center}
 #bar{height:6px;background:#0a0;transition:width .1s}
 #wrap{padding:10px}
 img{max-width:92vw;max-height:72vh;border-radius:8px;border:3px solid #333}
 .b{font-size:18px;padding:10px 20px;margin:6px;border:0;border-radius:8px;
    cursor:pointer;color:#fff}
 #keep{background:#1a8a1a}#nocube{background:#b8860b}#disc{background:#a31a1a}#back{background:#444}
 #status{font-size:16px;color:#aaa;margin-top:6px}
</style></head><body>
<div id="bar" style="width:0"></div>
<div id="wrap">
 <h3>Pseudo-label review</h3>
 <div><img id="im" src="/img/0"></div>
 <div id="status">1 / {{total}}</div>
 <button class="b" id="keep" onclick="act('keep')">Keep (K)</button>
 <button class="b" id="nocube" onclick="act('nocube')">No cube (N)</button>
 <button class="b" id="disc" onclick="act('discard')">Discard (D)</button>
 <button class="b" id="back" onclick="back()">Back</button>
</div>
<script>
 const total={{total}}; let i=0; const actions=[];
 function show(){
   document.getElementById('im').src='/img/'+i;
   document.getElementById('status').textContent=(i+1)+' / '+total;
   document.getElementById('bar').style.width=(100*i/total)+'%';
 }
 function act(a){ actions[i]=a; i++; if(i>=total) finish(); else show(); }
 function back(){ if(i>0){ i--; actions.length=i; show(); } }
 function finish(){
   fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
     body:JSON.stringify({actions})}).then(r=>r.json()).then(d=>{
     document.getElementById('wrap').innerHTML=
       '<h2>Done — keep '+d.keep+', no-cube '+d.nocube+', discard '+d.discard+'.</h2>';
     document.getElementById('bar').style.width='100%';
   });
 }
 document.onkeydown=e=>{
   const k=e.key;
   if(k==='k'||k==='K'||k==='ArrowRight'||k===' ') act('keep');
   else if(k==='n'||k==='N') act('nocube');
   else if(k==='d'||k==='D') act('discard');
   else if(k==='ArrowLeft') back();
 };
</script></body></html>"""


def create_app(frames_dir: Path, recover: bool) -> Flask:
    app = Flask(__name__)
    frames = _frames(frames_dir)

    @app.route("/")
    def index() -> str:
        return render_template_string(_PAGE, total=len(frames))

    @app.route("/img/<int:i>")
    def img(i: int) -> Response:
        return Response(_overlay(frames[i]), mimetype="image/jpeg")

    @app.route("/save", methods=["POST"])
    def save() -> Response:
        actions = request.get_json()["actions"]
        counts = {"keep": 0, "nocube": 0, "discard": 0}
        trash = frames_dir / "_discarded"
        for frame, action in zip(frames, actions, strict=False):
            counts[action] += 1
            label = frame.with_suffix(".txt")
            if recover:
                if action == "discard":
                    continue  # leave it in the discard pile
                moved = frames_dir.parent / frame.name
                frame.rename(moved)
                new_label = moved.with_suffix(".txt")
                if action == "nocube":
                    new_label.write_text("", encoding="utf-8")
                    label.unlink(missing_ok=True)
                else:
                    label.rename(new_label)
            elif action == "discard":
                trash.mkdir(exist_ok=True)
                frame.rename(trash / frame.name)
                label.rename(trash / label.name)
            elif action == "nocube":
                label.write_text("", encoding="utf-8")
        return jsonify(**counts)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Review pseudo-labels in a browser.")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Review a discard pile; Keep/No cube move frames back to the parent folder.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    app = create_app(Path(args.frames_dir), args.recover)
    print(f"[review_labels] open http://{args.host}:{args.port} — Ctrl+C to stop")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
