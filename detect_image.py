import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from utils import load_model, detect_people, draw_boxes_and_count  # utils NON espone gather_files_or_url

# ---------- helper locali per immagini ----------
ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def is_image(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTS

def is_url(s: str) -> bool:
    return isinstance(s, str) and s.lower().startswith(("http://", "https://"))

def base_name_for_url(url: str) -> str:
    name = os.path.basename(urlparse(url).path) or "image.jpg"
    if not os.path.splitext(name)[1]:
        name += ".jpg"
    return name

def imread_url(url: str, timeout=15):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def gather_files_or_url(input_arg: str):
    """Ritorna (lista_di_elementi, is_url_mode). Per URL singola ritorna [url], True."""
    if is_url(input_arg):
        return [input_arg], True
    p = Path(input_arg)
    if p.is_file() and is_image(p):
        return [str(p)], False
    if p.is_dir():
        files = [str(f) for f in p.iterdir() if f.is_file() and is_image(f)]
        return sorted(files), False
    return [], False
# ------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="immagine/cartella oppure URL")
    ap.add_argument("--out", default="outputs", help="cartella di output")
    ap.add_argument("--score_thr", type=float, default=0.6)
    ap.add_argument("--detector", choices=["fast","accurate"], default="fast",
                    help="Scegli l'algoritmo: fast=SSDLite (veloce), accurate=Faster R-CNN (più preciso)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files, is_url_mode = gather_files_or_url(args.input)
    if not files:
        print("[ERRORE] Nessuna immagine trovata o URL non valido.")
        return

    model, preprocess, det_name = load_model(detector=args.detector)
    print(f"[INFO] Detector attivo: {det_name}")

    for item in files:
        if is_url_mode:
            img = imread_url(item)
            out_name = base_name_for_url(item)
        else:
            img = cv2.imread(item)
            out_name = os.path.basename(item)

        if img is None:
            print(f"[ERRORE] Impossibile aprire immagine: {item}")
            continue

        boxes, scores = detect_people(model, preprocess, img, score_thr=args.score_thr)
        vis, count = draw_boxes_and_count(img, boxes, scores, args.score_thr)

        # salva con suffisso _det per non sovrascrivere
        root, ext = os.path.splitext(out_name)
        out_path = os.path.join(args.out, f"{root}_det{ext or '.jpg'}")
        cv2.imwrite(out_path, vis)
        print(f"[OK] {item} → Persone rilevate: {count}, salvato in {out_path}")

if __name__ == "__main__":
    main()
