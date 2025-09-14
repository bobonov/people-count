import argparse
import os
import cv2
from utils import load_model, detect_people, draw_boxes_and_count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="video file")
    ap.add_argument("--out", default="outputs", help="output directory")
    ap.add_argument("--score_thr", type=float, default=0.6)
    ap.add_argument("--device", default="cpu")  # mantenuto per compatibilità, non usato
    ap.add_argument("--detector", choices=["fast","accurate"], default="fast",
                    help="Scegli l'algoritmo: fast=SSDLite (veloce), accurate=Faster R-CNN (più preciso)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, preprocess, det_name = load_model(detector=args.detector)
    print(f"[INFO] Detector attivo: {det_name}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire il video: {args.input}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(args.out, os.path.basename(args.input))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores = detect_people(model, preprocess, frame, score_thr=args.score_thr)
        vis, _ = draw_boxes_and_count(frame, boxes, scores, args.score_thr)
        out.write(vis)

    cap.release()
    out.release()
    print(f"[OK] Video elaborato → {out_path}")

if __name__ == "__main__":
    main()
