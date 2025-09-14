# detect_cli.py (scelte numerate + fix chiusura webcam con 'q')
import os
import sys
import argparse
from typing import Optional, List, Tuple
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from utils import load_model, detect_people, draw_boxes_and_count

# ---------- helper locali (I/O immagini/URL) ----------
ALLOWED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

def is_image(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTS

def imread_unicode(path_str: str):
    img = cv2.imread(path_str)
    if img is not None:
        return img
    data = np.fromfile(path_str, dtype=np.uint8)  # fallback per path unicode su Windows
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imread_url(url: str, timeout=15):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def gather_files_or_url(input_arg: str) -> Tuple[List[str], bool]:
    if input_arg.startswith(("http://", "https://")):
        return [input_arg], True
    p = Path(input_arg)
    if p.is_file() and is_image(p):
        return [str(p)], False
    if p.is_dir():
        files = [str(f) for f in p.iterdir() if f.is_file() and is_image(f)]
        return sorted(files), False
    return [], False

def base_name_for_url(url: str):
    name = os.path.basename(urlparse(url).path) or "image.jpg"
    if not os.path.splitext(name)[1]:
        name += ".jpg"
    return name
# ------------------------------------------------------

# ---------- prompt numerati ----------
def prompt_choice(title: str, options: List[str], default_index: int = 0) -> str:
    """
    Stampa un menu numerato e ritorna l'opzione scelta.
    Accetta: numero (1..N), nome opzione (case-insensitive), oppure Invio per default.
    """
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        star = "  (default)" if (i - 1) == default_index else ""
        print(f"  {i}) {opt}{star}")
    while True:
        raw = input(f"Seleziona [1-{len(options)}] o premi Invio per {options[default_index]}: ").strip()
        if not raw:
            return options[default_index]
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        # consenti anche per nome
        for opt in options:
            if raw.lower() == opt.lower():
                return opt
        print("Scelta non valida. Riprova.")

def prompt_text(label: str, default: Optional[str] = None) -> str:
    dflt = f" (default: {default})" if default is not None else ""
    val = input(f"{label}{dflt}: ").strip()
    return default if (not val and default is not None) else val
# -------------------------------------

def run_images(detector: str, input_path: str, out_dir: str, score_thr: float):
    os.makedirs(out_dir, exist_ok=True)
    files, is_url_mode = gather_files_or_url(input_path)
    if not files:
        print("[ERRORE] Nessuna immagine trovata o URL non valido.")
        return
    model, preprocess, det_name = load_model(detector=detector)
    print(f"[INFO] Detector attivo: {det_name}")
    for item in files:
        if is_url_mode:
            img = imread_url(item)
            name = base_name_for_url(item)
        else:
            img = imread_unicode(item)
            name = Path(item).name

        if img is None:
            print(f"[WARN] Impossibile leggere: {item}")
            continue

        boxes, scores = detect_people(model, preprocess, img, score_thr=score_thr)
        vis, count = draw_boxes_and_count(img, boxes, scores, score_thr)
        stem = Path(name).stem
        out_img = os.path.join(out_dir, f"{stem}_det.jpg")
        cv2.imwrite(out_img, vis)
        print(f"[OK] {name} → Persone: {count} | salvato: {out_img}")

def run_video(detector: str, video_path: str, out_dir: str, score_thr: float):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire il video: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(out_dir, Path(video_path).stem + "_out.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    model, preprocess, det_name = load_model(detector=detector)
    print(f"[INFO] Detector attivo: {det_name}")

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, scores = detect_people(model, preprocess, frame, score_thr=score_thr)
        vis, _ = draw_boxes_and_count(frame, boxes, scores, score_thr)
        writer.write(vis)
        i += 1
        if i % 50 == 0:
            print(f"[INFO] Frame elaborati: {i}")

    cap.release()
    writer.release()
    print(f"[OK] Video elaborato → {out_path}")

def run_webcam(detector: str, score_thr: float):
    # su Windows CAP_DSHOW evita delay; namedWindow garantisce focus esplicito
    win_name = "People Count - Webcam (premi q / Esc per uscire)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if os.name == "nt":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERRORE] Impossibile aprire la webcam.")
        return

    model, preprocess, det_name = load_model(detector=detector)
    print(f"[INFO] Detector attivo: {det_name}")
    print("[INFO] Premi 'q' o 'Esc' per uscire (la finestra deve avere il focus).")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores = detect_people(model, preprocess, frame, score_thr=score_thr)
        vis, count = draw_boxes_and_count(frame, boxes, scores, score_thr)

        cv2.imshow(win_name, vis)
        key = cv2.waitKey(1)
        if key != -1:
            k = key & 0xFF
            if k in (ord('q'), ord('Q'), 27):  # 27 = Esc
                break

    cap.release()
    cv2.destroyWindow(win_name)
    cv2.waitKey(1)  # flush eventi su alcune piattaforme

def main():
    parser = argparse.ArgumentParser(description="People Count CLI guidata (immagini / video / webcam) con scelte numerate.")
    parser.add_argument("--mode", choices=["image", "video", "webcam"], help="Modalità di esecuzione")
    parser.add_argument("--detector", choices=["fast", "accurate"], help="Algoritmo: fast=SSDLite, accurate=Faster R-CNN")
    parser.add_argument("--input", help="Percorso immagine/cartella/URL (image) o file video (video)")
    parser.add_argument("--out", default="outputs", help="Cartella output (default: outputs)")
    parser.add_argument("--score_thr", type=float, default=0.6, help="Soglia confidenza (default: 0.6)")
    args = parser.parse_args()

    # menu numerati se mancano parametri
    mode = args.mode or prompt_choice("Seleziona modalità", ["image", "video", "webcam"], default_index=0)
    detector = args.detector or prompt_choice("Scegli algoritmo", ["fast", "accurate"], default_index=0)
    score_thr = args.score_thr
    out_dir = args.out

    if mode == "image":
        input_path = args.input or prompt_text("Percorso immagine/cartella o URL", default="./imgs")
        print(f"\n[RIEPILOGO]\n  Modalità : {mode}\n  Detector : {detector}\n  Input    : {input_path}\n  Output   : {out_dir}\n  Soglia   : {score_thr}\n")
        run_images(detector=detector, input_path=input_path, out_dir=out_dir, score_thr=score_thr)
    elif mode == "video":
        video_path = args.input or prompt_text("Percorso file video", default="./video.mp4")
        print(f"\n[RIEPILOGO]\n  Modalità : {mode}\n  Detector : {detector}\n  Video    : {video_path}\n  Output   : {out_dir}\n  Soglia   : {score_thr}\n")
        run_video(detector=detector, video_path=video_path, out_dir=out_dir, score_thr=score_thr)
    else:  # webcam
        print(f"\n[RIEPILOGO]\n  Modalità : {mode}\n  Detector : {detector}\n  Soglia   : {score_thr}\n  (live, nessun file di output)\n")
        run_webcam(detector=detector, score_thr=score_thr)

if __name__ == "__main__":
    try:
        import cv2  # noqa: F401
    except Exception:
        print("[ERRORE] OpenCV non è installato. Installa con: pip install opencv-python")
        sys.exit(1)
    main()
