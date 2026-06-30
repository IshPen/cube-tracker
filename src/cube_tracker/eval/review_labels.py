"""Browser tool to verify Grounding DINO pseudo-labels before fine-tuning (M4 hardening).

Shows each real frame with its auto-label box (or a "NO CUBE" banner for hard negatives) one at
a time; keep the good ones and discard the wrong ones (a box on the wrong object, or a missed
cube wrongly marked empty). Discarded frames are moved to ``_discarded/`` so nothing is lost.
Keyboard: K / → / Space = keep, D / ← skip-back. Needs the ``demo`` extra (flask).

    python -m cube_tracker.eval.review_labels --frames-dir data/real_frames
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
 img{max-width:92vw;max-height:74vh;border-radius:8px;border:3px solid #333}
 .b{font-size:18px;padding:10px 22px;margin:8px;border:0;border-radius:8px;
    cursor:pointer;color:#fff}
 #keep{background:#1a8a1a}#disc{background:#a31a1a}#back{background:#444}
 #status{font-size:16px;color:#aaa;margin-top:6px}
</style></head><body>
<div id="bar" style="width:0"></div>
<div id="wrap">
 <h3>Pseudo-label review — keep good boxes, discard wrong/missed</h3>
 <div><img id="im" src="/img/0"></div>
 <div id="status">1 / {{total}}</div>
 <button class="b" id="keep" onclick="decide(true)">Keep (K)</button>
 <button class="b" id="disc" onclick="decide(false)">Discard (D)</button>
 <button class="b" id="back" onclick="back()">Back</button>
</div>
<script>
 const total={{total}}; let i=0; const discard=[];
 function show(){
   document.getElementById('im').src='/img/'+i;
   document.getElementById('status').textContent=
     (i+1)+' / '+total+' (discarded: '+discard.length+')';
   document.getElementById('bar').style.width=(100*i/total)+'%';
 }
 function decide(keep){ if(!keep) discard.push(i); i++; if(i>=total) finish(); else show(); }
 function back(){
   if(i>0){ i--; const k=discard.indexOf(i); if(k>=0) discard.splice(k,1); show(); }
 }
 function finish(){
   fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
     body:JSON.stringify({discard})}).then(r=>r.json()).then(d=>{
     document.getElementById('wrap').innerHTML=
       '<h2>Done — kept '+d.kept+', discarded '+d.discarded+'.</h2>';
     document.getElementById('bar').style.width='100%';
   });
 }
 document.onkeydown=e=>{
   if(e.key==='k'||e.key==='K'||e.key==='ArrowRight'||e.key===' ') decide(true);
   else if(e.key==='d'||e.key==='D') decide(false);
   else if(e.key==='ArrowLeft') back();
 };
</script></body></html>"""


def create_app(frames_dir: Path) -> Flask:
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
        discard = set(request.get_json()["discard"])
        trash = frames_dir / "_discarded"
        trash.mkdir(exist_ok=True)
        for index_ in discard:
            frame = frames[index_]
            frame.rename(trash / frame.name)
            frame.with_suffix(".txt").rename(trash / f"{frame.stem}.txt")
        return jsonify(kept=len(frames) - len(discard), discarded=len(discard))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Review pseudo-labels in a browser.")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    app = create_app(Path(args.frames_dir))
    print(f"[review_labels] open http://{args.host}:{args.port} — Ctrl+C to stop")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
