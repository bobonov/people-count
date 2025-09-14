# detect_gui.py
# GUI didattica per confronto SSDLite (fast) vs Faster R-CNN (accurate)
# - Immagini: fast / accurate / BOTH (confronto side-by-side + tempi) da FILE o URL (http/https)
# - Video: fast / accurate (da file)
# - Webcam: fast / accurate
#
# Dipendenze: tkinter, pillow, opencv-python, numpy, requests
# Moduli progetto: utils.py con load_model, detect_people, draw_boxes_and_count

import os
import time
import threading
from pathlib import Path
from typing import Dict, Tuple, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
import requests

from utils import load_model, detect_people, draw_boxes_and_count


# ----------------------------- utility immagini -----------------------------
ALLOWED_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def is_url(s: str) -> bool:
    return isinstance(s, str) and s.lower().startswith(("http://", "https://"))

def imread_url(url: str, timeout: int = 15) -> Optional[np.ndarray]:
    """Scarica un'immagine da URL e la decodifica in BGR (cv2)."""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERRORE] Download URL fallito: {e}")
        return None

def imread_unicode(path_str: str) -> Optional[np.ndarray]:
    """Legge file immagine con fallback per path unicode su Windows."""
    img = cv2.imread(path_str)
    if img is not None:
        return img
    try:
        data = np.fromfile(path_str, dtype=np.uint8)  # fallback unicode
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERRORE] Lettura file fallita: {e}")
        return None

def cv2_to_tk(img_bgr, max_side: int = 640) -> Optional[ImageTk.PhotoImage]:
    """Converte un'immagine BGR (numpy) in PhotoImage ridimensionando per stare nella GUI."""
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        nh, nw = int(h * scale), int(w * scale)
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(img_rgb)
    return ImageTk.PhotoImage(pil)

def draw_title_bar(image_bgr, title: str):
    """Disegna una barra in alto con testo (per indicare detector e tempo)."""
    if image_bgr is None:
        return image_bgr
    img = image_bgr.copy()
    h, w = img.shape[:2]
    bar_h = max(28, h // 20)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, title, (12, int(bar_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img
# ---------------------------------------------------------------------------


class PeopleCountApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("People Count GUI — Confronto detector (File/URL)")
        self.geometry("1180x760")

        # Stato condiviso
        self.score_var = tk.DoubleVar(value=0.60)

        # Detector per ciascun tab
        self.detector_img_var = tk.StringVar(value="fast")   # image: fast / accurate / both
        self.detector_vid_var = tk.StringVar(value="fast")   # video: fast / accurate
        self.detector_cam_var = tk.StringVar(value="fast")   # webcam: fast / accurate

        # Cache modelli: {"fast": (model, preprocess), "accurate": (...)}
        self.models: Dict[str, Tuple[object, object]] = {}

        # Riferimenti alle immagini Tk per evitare garbage collection
        self.tk_img_left: Optional[ImageTk.PhotoImage] = None
        self.tk_img_right: Optional[ImageTk.PhotoImage] = None
        self.tk_img_single: Optional[ImageTk.PhotoImage] = None

        # Webcam state
        self.cam_running = False
        self.cam_thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None

        self._build_ui()

    # ------------------------------- UI BUILD --------------------------------
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        # Tabs
        self.tab_img = ttk.Frame(nb)
        self.tab_video = ttk.Frame(nb)
        self.tab_cam = ttk.Frame(nb)
        nb.add(self.tab_img, text="Immagine (File/URL)")
        nb.add(self.tab_video, text="Video")
        nb.add(self.tab_cam, text="Webcam")

        # -------- Tab Immagine --------
        top_img = tk.Frame(self.tab_img)
        top_img.pack(fill="x", padx=8, pady=8)

        tk.Label(top_img, text="Immagine o URL:").pack(side="left")
        self.img_path_var = tk.StringVar()
        tk.Entry(top_img, textvariable=self.img_path_var, width=70).pack(side="left", padx=6)
        tk.Button(top_img, text="Sfoglia…", command=self.on_browse_image).pack(side="left", padx=4)

        # soglia
        ctrl_img = tk.Frame(self.tab_img)
        ctrl_img.pack(fill="x", padx=8, pady=(0, 6))
        tk.Label(ctrl_img, text="Soglia confidenza:").pack(side="left")
        tk.Scale(ctrl_img, variable=self.score_var, from_=0.1, to=0.95, resolution=0.05,
                 orient="horizontal", length=260).pack(side="left", padx=8)

        # Detector anche per immagini — con BOTH
        det_img = tk.Frame(self.tab_img)
        det_img.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(det_img, text="Detector (immagini):").pack(side="left")
        ttk.Combobox(det_img,
                     values=["fast", "accurate", "both"],
                     textvariable=self.detector_img_var,
                     width=10, state="readonly").pack(side="left", padx=6)
        tk.Button(det_img, text="Rileva", command=self.on_detect_image, width=14).pack(side="left", padx=6)

        # area anteprima immagini (sinistra/destra)
        self.view_img = tk.Frame(self.tab_img)
        self.view_img.pack(fill="both", expand=True, padx=8, pady=8)

        self.left_panel = tk.Label(self.view_img, bd=1, relief="sunken", width=64, height=24, bg="#222")
        self.right_panel = tk.Label(self.view_img, bd=1, relief="sunken", width=64, height=24, bg="#222")
        self.left_panel.pack(side="left", expand=True, fill="both", padx=(0, 4))
        self.right_panel.pack(side="left", expand=True, fill="both", padx=(4, 0))

        self.info_img = tk.Label(self.tab_img, text="", anchor="w", justify="left")
        self.info_img.pack(fill="x", padx=8, pady=(0, 8))

        # -------- Tab Video --------
        top_vid = tk.Frame(self.tab_video)
        top_vid.pack(fill="x", padx=8, pady=8)

        tk.Label(top_vid, text="Video:").pack(side="left")
        self.vid_path_var = tk.StringVar()
        tk.Entry(top_vid, textvariable=self.vid_path_var, width=70).pack(side="left", padx=6)
        tk.Button(top_vid, text="Sfoglia…", command=self.on_browse_video).pack(side="left", padx=4)

        ctrl_vid = tk.Frame(self.tab_video); ctrl_vid.pack(fill="x", padx=8, pady=(0, 6))
        tk.Label(ctrl_vid, text="Soglia confidenza:").pack(side="left")
        tk.Scale(ctrl_vid, variable=self.score_var, from_=0.1, to=0.95, resolution=0.05,
                 orient="horizontal", length=260).pack(side="left", padx=8)

        det_vid = tk.Frame(self.tab_video); det_vid.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(det_vid, text="Detector (video):").pack(side="left")
        ttk.Combobox(det_vid, values=["fast", "accurate"], textvariable=self.detector_vid_var,
                     width=10, state="readonly").pack(side="left", padx=6)
        tk.Button(det_vid, text="Elabora Video", command=self.on_process_video).pack(side="left", padx=6)

        self.info_vid = tk.Label(self.tab_video, text="", anchor="w", justify="left")
        self.info_vid.pack(fill="x", padx=8, pady=(0, 8))

        # -------- Tab Webcam --------
        ctrl_cam = tk.Frame(self.tab_cam); ctrl_cam.pack(fill="x", padx=8, pady=(8, 6))
        tk.Label(ctrl_cam, text="Soglia confidenza:").pack(side="left")
        tk.Scale(ctrl_cam, variable=self.score_var, from_=0.1, to=0.95, resolution=0.05,
                 orient="horizontal", length=260).pack(side="left", padx=8)

        det_cam = tk.Frame(self.tab_cam); det_cam.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(det_cam, text="Detector (webcam):").pack(side="left")
        ttk.Combobox(det_cam, values=["fast", "accurate"], textvariable=self.detector_cam_var,
                     width=10, state="readonly").pack(side="left", padx=6)
        tk.Button(det_cam, text="Avvia", command=self.on_start_cam).pack(side="left", padx=6)
        tk.Button(det_cam, text="Stop", command=self.on_stop_cam).pack(side="left", padx=6)

        self.cam_panel = tk.Label(self.tab_cam, bd=1, relief="sunken", bg="#222")
        self.cam_panel.pack(fill="both", expand=True, padx=8, pady=8)
        self.info_cam = tk.Label(self.tab_cam, text="", anchor="w", justify="left")
        self.info_cam.pack(fill="x", padx=8, pady=(0, 8))

        # scorciatoie veloci per cambiare detector immagini
        self.bind("<Control-Key-1>", lambda e: self._quick_set_img_detector("fast"))
        self.bind("<Control-Key-2>", lambda e: self._quick_set_img_detector("accurate"))
        self.bind("<Control-Key-3>", lambda e: self._quick_set_img_detector("both"))

    # ----------------------------- MODEL HANDLING -----------------------------
    def get_model(self, det: str):
        """Ritorna (model, preprocess) per il detector richiesto, caricandolo se mancante."""
        if det not in self.models:
            model, preprocess, _ = load_model(detector=det)
            self.models[det] = (model, preprocess)
        return self.models[det]

    # ------------------------------- IMAGE FLOW -------------------------------
    def on_browse_image(self):
        fpath = filedialog.askopenfilename(
            title="Seleziona immagine",
            filetypes=[("Immagini", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp"), ("Tutti i file", "*.*")]
        )
        if fpath:
            self.img_path_var.set(fpath)

    def _quick_set_img_detector(self, name: str):
        if name in ("fast", "accurate", "both"):
            self.detector_img_var.set(name)

    def _load_image(self, src: str) -> Optional[np.ndarray]:
        """Carica immagine da file locale o URL."""
        if not src:
            messagebox.showwarning("Attenzione", "Inserisci un percorso o un URL di immagine.")
            return None
        if is_url(src):
            img = imread_url(src)
            if img is None:
                messagebox.showerror("Errore", "Download/decodifica da URL fallito.")
            return img
        # file locale
        if not Path(src).exists():
            messagebox.showerror("Errore", f"File non trovato:\n{src}")
            return None
        if not src.lower().endswith(ALLOWED_IMG_EXTS):
            # proviamo comunque a leggerla
            pass
        img = imread_unicode(src)
        if img is None:
            messagebox.showerror("Errore", "Formato non supportato o file corrotto.")
        return img

    def on_detect_image(self):
        src = self.img_path_var.get().strip()
        img = self._load_image(src)
        if img is None:
            return

        thr = float(self.score_var.get())
        choice = self.detector_img_var.get()
        self.info_img.config(text="")

        if choice in ("fast", "accurate"):
            det = choice
            model, preprocess = self.get_model(det)
            t0 = time.perf_counter()
            boxes, scores = detect_people(model, preprocess, img, score_thr=thr)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            vis, count = draw_boxes_and_count(img.copy(), boxes, scores, thr)
            title = f"{det.upper()} - Persone: {count} - {dt_ms:.1f} ms"
            vis = draw_title_bar(vis, title)

            # mostra singola immagine a sinistra e pulisci destra
            self.tk_img_left = cv2_to_tk(vis, max_side=700)
            self.left_panel.config(image=self.tk_img_left)
            self.right_panel.config(image="")

            src_txt = src if is_url(src) else Path(src).name
            self.info_img.config(text=f"{src_txt} | {title}")

        else:  # BOTH
            # FAST
            model_f, pre_f = self.get_model("fast")
            t0 = time.perf_counter()
            boxes_f, scores_f = detect_people(model_f, pre_f, img, score_thr=thr)
            ms_fast = (time.perf_counter() - t0) * 1000.0
            vis_f, count_f = draw_boxes_and_count(img.copy(), boxes_f, scores_f, thr)
            vis_f = draw_title_bar(vis_f, f"FAST (SSDLite) - Persone: {count_f} - {ms_fast:.1f} ms")

            # ACCURATE
            model_a, pre_a = self.get_model("accurate")
            t0 = time.perf_counter()
            boxes_a, scores_a = detect_people(model_a, pre_a, img, score_thr=thr)
            ms_acc = (time.perf_counter() - t0) * 1000.0
            vis_a, count_a = draw_boxes_and_count(img.copy(), boxes_a, scores_a, thr)
            vis_a = draw_title_bar(vis_a, f"ACCURATE (Faster R-CNN) - Persone: {count_a} - {ms_acc:.1f} ms")

            # mostra side-by-side
            self.tk_img_left = cv2_to_tk(vis_f, max_side=700)
            self.tk_img_right = cv2_to_tk(vis_a, max_side=700)
            self.left_panel.config(image=self.tk_img_left)
            self.right_panel.config(image=self.tk_img_right)

            src_txt = src if is_url(src) else Path(src).name
            self.info_img.config(
                text=f"{src_txt} | Confronto — FAST: {ms_fast:.1f} ms (persone {count_f}) | "
                     f"ACCURATE: {ms_acc:.1f} ms (persone {count_a})"
            )

    # ------------------------------- VIDEO FLOW -------------------------------
    def on_browse_video(self):
        fpath = filedialog.askopenfilename(
            title="Seleziona video",
            filetypes=[("Video", "*.mp4;*.mov;*.avi;*.mkv;*.m4v;*.wmv"), ("Tutti i file", "*.*")]
        )
        if fpath:
            self.vid_path_var.set(fpath)

    def on_process_video(self):
        path = self.vid_path_var.get().strip()
        if not path:
            messagebox.showwarning("Attenzione", "Seleziona un video.")
            return
        if not Path(path).exists():
            messagebox.showerror("Errore", f"File non trovato:\n{path}")
            return

        det = self.detector_vid_var.get()
        thr = float(self.score_var.get())
        try:
            model, preprocess = self.get_model(det)
        except Exception as e:
            messagebox.showerror("Errore", f"Caricamento modello fallito: {e}")
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Errore", "Impossibile aprire il video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
        out_path = out_dir / (Path(path).stem + f"_{det}.mp4")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        self.info_vid.config(text=f"[{det}] Elaborazione in corso…")

        n = 0
        t0_total = time.perf_counter()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            boxes, scores = detect_people(model, preprocess, frame, score_thr=thr)
            vis, _ = draw_boxes_and_count(frame, boxes, scores, thr)
            writer.write(vis)
            n += 1
            if n % 50 == 0:
                self.info_vid.config(text=f"[{det}] Frame elaborati: {n}")
                self.update_idletasks()

        elapsed_s = time.perf_counter() - t0_total
        cap.release()
        writer.release()

        self.info_vid.config(text=f"[{det}] Finito: {n} frame → {out_path} (t={elapsed_s:.1f}s)")

    # ------------------------------- WEBCAM FLOW ------------------------------
    def on_start_cam(self):
        if self.cam_running:
            return
        det = self.detector_cam_var.get()
        thr = float(self.score_var.get())
        try:
            model, preprocess = self.get_model(det)
        except Exception as e:
            messagebox.showerror("Errore", f"Caricamento modello fallito: {e}")
            return

        # apro webcam
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Errore", "Impossibile aprire la webcam.")
            return

        self.cam_running = True
        self.info_cam.config(text=f"Webcam attiva — detector: {det} — premi Stop per terminare.")

        def loop():
            while self.cam_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                t0 = time.perf_counter()
                boxes, scores = detect_people(model, preprocess, frame, score_thr=thr)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                vis, count = draw_boxes_and_count(frame, boxes, scores, thr)
                vis = draw_title_bar(vis, f"{det.upper()} — Persone: {count} — {dt_ms:.1f} ms")

                tk_img = cv2_to_tk(vis, max_side=900)
                self.tk_img_single = tk_img
                self.cam_panel.config(image=tk_img)
                self.info_cam.config(text=f"Detector: {det} — {dt_ms:.1f} ms — Persone: {count}")

            if self.cap:
                self.cap.release()
            self.cap = None

        self.cam_thread = threading.Thread(target=loop, daemon=True)
        self.cam_thread.start()

    def on_stop_cam(self):
        self.cam_running = False
        self.info_cam.config(text="Webcam fermata.")
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.cam_panel.config(image="")
        self.tk_img_single = None


if __name__ == "__main__":
    try:
        import cv2  # noqa: F401
    except Exception:
        messagebox = None
        print("[ERRORE] OpenCV non è installato. Installa con: pip install opencv-python")
        raise
    app = PeopleCountApp()
    app.mainloop()
