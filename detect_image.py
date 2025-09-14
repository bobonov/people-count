import argparse
import os
import cv2
from utils import load_model, gather_files_or_url, detect_people, draw_boxes_and_count

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

    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"[ERRORE] Impossibile aprire immagine: {f}")
            continue

        boxes, scores = detect_people(model, preprocess, img, score_thr=args.score_thr)
        vis, count = draw_boxes_and_count(img, boxes, scores, args.score_thr)

        out_path = os.path.join(args.out, os.path.basename(f))
        cv2.imwrite(out_path, vis)
        print(f"[OK] {f} → Persone rilevate: {count}, salvato in {out_path}")

if __name__ == "__main__":
    main()
