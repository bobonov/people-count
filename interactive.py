import argparse
import os
import cv2
from utils import load_model, detect_people, draw_boxes_and_count

def main():
    ap = argparse.ArgumentParser(description="People Counting - Interfaccia interattiva")
    ap.add_argument("--out", default="outputs", help="cartella di output")
    ap.add_argument("--score_thr", type=float, default=0.6, help="soglia confidenza")
    ap.add_argument("--device", default="cpu", help="(non usato)")
    ap.add_argument("--detector", choices=["fast","accurate"], default="fast",
                    help="Scegli l'algoritmo: fast=SSDLite (veloce), accurate=Faster R-CNN (più preciso)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, preprocess, det_name = load_model(detector=args.detector)
    print(f"[INFO] Detector attivo: {det_name}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRORE] Impossibile aprire la webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores = detect_people(model, preprocess, frame, score_thr=args.score_thr)
        vis, count = draw_boxes_and_count(frame, boxes, scores, args.score_thr)

        cv2.imshow("Interactive - Rilevamento Persone", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
