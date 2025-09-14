import os, re, time
import cv2, numpy as np, requests, torch
from urllib.parse import urlparse
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from utils import load_model  # ora supporta detector="fast"/"accurate"

# --- utilità immagine ---
def imread_unicode(path_str: str):
    img = cv2.imread(path_str)
    if img is not None:
        return img
    data = np.fromfile(path_str, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imread_url_image(url: str, timeout=15):
    r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def name_from_url(url: str, default="image.jpg"):
    name = os.path.basename(urlparse(url).path) or default
    if not os.path.splitext(name)[1]:
        name += ".jpg"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name

class PeopleCountingApp(tk.Tk):
    def __init__(self, max_w=1000, max_h=700):
        super().__init__()
        self.title("People Counting - GUI (CPU)")
        self.max_w, self.max_h = max_w, max_h

        # stato immagine
        self.img_bgr = None
        self.img_disp = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.boxes_all = None
        self.scores_all = None
        self.base_name = None

        # stato video
        self.cap = None
        self.writer = None
        self.running = False
        self.video_src_kind = tk.StringVar(value="webcam")  # 'file'|'url'|'webcam'
        self.save_video_var = tk.BooleanVar(value=False)
        self.webcam_index_var = tk.IntVar(value=0)
        self.cam_w_var = tk.IntVar(value=640)
        self.cam_h_var = tk.IntVar(value=480)
        self.proc_max_side_var = tk.IntVar(value=720)   # riduci per più FPS
        self.frame_skip_var = tk.IntVar(value=2)        # inferenza ogni N frame

        # detector selection
        self.detector_var = tk.StringVar(value="fast")  # "fast" (SSDLite) | "accurate" (Faster R-CNN)
        self.model, self.preprocess, self.detector_name = load_model(detector=self.detector_var.get())

        # UI
        self._build_ui()

    # ------------------------ UI ------------------------
    def _build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        # --- TAB IMMAGINE ---
        tab_img = ttk.Frame(notebook); notebook.add(tab_img, text="Immagine")
        top = tk.Frame(tab_img); top.pack(fill="x", padx=8, pady=8)
        tk.Label(top, text="Percorso locale o URL:").pack(side="left")
        self.entry_img = tk.Entry(top, width=70); self.entry_img.pack(side="left", padx=6)
        tk.Button(top, text="Sfoglia…", command=self.on_browse_image).pack(side="left", padx=4)
        tk.Button(top, text="Carica", command=self.on_load_image).pack(side="left", padx=4)
        tk.Button(top, text="Salva annotata", command=self.on_save_image).pack(side="left", padx=4)

        ctrl_img = tk.Frame(tab_img); ctrl_img.pack(fill="x", padx=8, pady=(0,8))
        self.threshold = tk.DoubleVar(value=0.6)
        tk.Label(ctrl_img, text="Threshold:").pack(side="left")
        self.slider = tk.Scale(ctrl_img, from_=0.1, to=0.95, resolution=0.01,
                               orient="horizontal", variable=self.threshold,
                               command=self.on_threshold_change_image, length=300)
        self.slider.pack(side="left", padx=6)
        self.count_var = tk.StringVar(value="COUNT: -")
        tk.Label(ctrl_img, textvariable=self.count_var, font=("Arial", 12, "bold")).pack(side="left", padx=12)

        # --- TAB VIDEO ---
        tab_vid = ttk.Frame(notebook); notebook.add(tab_vid, text="Video")

        src_box = tk.LabelFrame(tab_vid, text="Sorgente video"); src_box.pack(fill="x", padx=8, pady=8)
        for lbl, val in (("File", "file"), ("URL", "url"), ("Webcam", "webcam")):
            tk.Radiobutton(src_box, text=lbl, value=val, variable=self.video_src_kind).pack(side="left", padx=6)

        self.entry_video = tk.Entry(src_box, width=60); self.entry_video.pack(side="left", padx=6)
        tk.Button(src_box, text="Sfoglia…", command=self.on_browse_video).pack(side="left", padx=4)

        tk.Label(src_box, text="Webcam index:").pack(side="left", padx=(12,2))
        tk.Spinbox(src_box, from_=0, to=9, width=3, textvariable=self.webcam_index_var).pack(side="left")

        # Opzioni performance
        opt = tk.LabelFrame(tab_vid, text="Opzioni performance"); opt.pack(fill="x", padx=8, pady=(0,8))
        tk.Label(opt, text="Detector:").pack(side="left")
        ttk.Combobox(opt, values=["fast","accurate"], textvariable=self.detector_var, width=10, state="readonly").pack(side="left", padx=6)
        tk.Button(opt, text="Applica detector", command=self.on_apply_detector).pack(side="left", padx=6)

        tk.Label(opt, text="Frame skip (N):").pack(side="left", padx=(12,2))
        tk.Spinbox(opt, from_=1, to=10, width=4, textvariable=self.frame_skip_var).pack(side="left")

        tk.Label(opt, text="Max lato inferenza:").pack(side="left", padx=(12,2))
        tk.Spinbox(opt, from_=400, to=1280, increment=20, width=6, textvariable=self.proc_max_side_var).pack(side="left")

        tk.Label(opt, text="Webcam W×H:").pack(side="left", padx=(12,2))
        tk.Spinbox(opt, from_=320, to=1920, increment=40, width=5, textvariable=self.cam_w_var).pack(side="left")
        tk.Label(opt, text="×").pack(side="left")
        tk.Spinbox(opt, from_=240, to=1080, increment=30, width=5, textvariable=self.cam_h_var).pack(side="left")

        tk.Checkbutton(opt, text="Salva video annotato", variable=self.save_video_var).pack(side="left", padx=12)

        tk.Button(opt, text="Avvia", command=self.on_start_video).pack(side="left", padx=6)
        tk.Button(opt, text="Stop", command=self.on_stop_video).pack(side="left", padx=6)

        self.fps_var = tk.StringVar(value="FPS: -")
        tk.Label(opt, textvariable=self.fps_var).pack(side="left", padx=12)

        # Canvas condiviso
        self.canvas = tk.Label(self, bd=2, relief="groove"); self.canvas.pack(padx=8, pady=8)

        # Status
        self.status = tk.StringVar(value=f"Pronto. Detector: {self.detector_name}")
        tk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", padx=8, pady=(0,8))

        self.bind("<Return>", lambda e: self.on_load_image())

    # -------------------- Detector switch --------------------
    def on_apply_detector(self):
        self.on_stop_video()
        self.model, self.preprocess, self.detector_name = load_model(detector=self.detector_var.get())
        self.status.set(f"Detector attivo: {self.detector_name}")

    # -------------------- IMMAGINE --------------------
    def on_browse_image(self):
        path = filedialog.askopenfilename(title="Seleziona immagine",
            filetypes=[("Immagini","*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp")])
        if path:
            self.entry_img.delete(0, tk.END); self.entry_img.insert(0, path)

    def on_load_image(self):
        self.on_stop_video()
        s = self.entry_img.get().strip()
        if not s:
            messagebox.showwarning("Attenzione", "Inserire un percorso o una URL."); return
        try:
            if s.startswith(("http://","https://")):
                img = imread_url_image(s); self.base_name = name_from_url(s)
            else:
                img = imread_unicode(s); self.base_name = os.path.basename(s)
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile leggere l'immagine:\n{e}"); return
        if img is None:
            messagebox.showerror("Errore", "Lettura immagine fallita."); return

        self.img_bgr = img
        self.status.set("Eseguo il detector (CPU)…"); self.update_idletasks()

        try:
            img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            tensor = self.preprocess(pil).to("cpu")
            with torch.no_grad():
                out = self.model([tensor])[0]
            boxes  = out["boxes"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            person = (labels == 1)
            self.boxes_all = boxes[person]; self.scores_all = scores[person]
        except Exception as e:
            messagebox.showerror("Errore", f"Inference fallita:\n{e}"); return

        self._prepare_preview_image()
        self._render_image_with_threshold()
        self.status.set(f"Pronto (immagine). Detector: {self.detector_name}")

    def on_threshold_change_image(self, _=None):
        if self.img_bgr is None or self.boxes_all is None: return
        self._render_image_with_threshold()

    def on_save_image(self):
        if self.img_bgr is None or self.boxes_all is None:
            messagebox.showwarning("Attenzione", "Carica prima un'immagine."); return
        thr = float(self.threshold.get())
        keep = self.scores_all >= thr
        boxes = self.boxes_all[keep]
        out = self.img_bgr.copy()
        for (x1,y1,x2,y2), s in zip(boxes, self.scores_all[keep]):
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(out, f"person {s:.2f}", (x1, max(y1-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)
        count = len(boxes)
        cv2.putText(out, f"COUNT: {count} (thr={thr:.2f})", (8,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        stem, _ = os.path.splitext(self.base_name or "image")
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite(os.path.join("outputs", f"{stem}_det_thr{thr:.2f}.jpg"), out)
        messagebox.showinfo("Salvato", "Immagine annotata salvata in outputs/")

    def _prepare_preview_image(self):
        h, w = self.img_bgr.shape[:2]
        scale = min(self.max_w / w, self.max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        self.scale_x = new_w / w; self.scale_y = new_h / h
        self.img_disp = cv2.resize(self.img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _render_image_with_threshold(self):
        thr = float(self.threshold.get())
        keep = self.scores_all >= thr
        boxes = self.boxes_all[keep]; scores = self.scores_all[keep]
        disp = self.img_disp.copy()
        for (x1,y1,x2,y2), s in zip(boxes, scores):
            x1 = int(x1 * self.scale_x); y1 = int(y1 * self.scale_y)
            x2 = int(x2 * self.scale_x); y2 = int(y2 * self.scale_y)
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(disp, f"person {s:.2f}", (x1, max(y1-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)
        count = len(boxes)
        cv2.putText(disp, f"COUNT: {count} (thr={thr:.2f})", (8,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(image=tk_img); self.canvas.image = tk_img
        self.count_var.set(f"COUNT: {count}")

    # ---------------------- VIDEO -----------------------
    def on_browse_video(self):
        if self.video_src_kind.get() != "file":
            messagebox.showinfo("Info", "Seleziona 'File' come sorgente per usare 'Sfoglia…'.")
            return
        path = filedialog.askopenfilename(title="Seleziona video",
            filetypes=[("Video","*.mp4;*.mov;*.avi;*.mkv;*.webm;*.m4v;*.mpg;*.mpeg")])
        if path:
            self.entry_video.delete(0, tk.END); self.entry_video.insert(0, path)

    def on_start_video(self):
        self.on_stop_video()  # chiudi eventuali risorse precedenti
        kind = self.video_src_kind.get()
        src = self.entry_video.get().strip()

        if kind == "file":
            if not src: return messagebox.showwarning("Attenzione", "Seleziona un file video.")
            cap = cv2.VideoCapture(src); base_name = os.path.basename(src)
        elif kind == "url":
            if not src: return messagebox.showwarning("Attenzione", "Inserisci una URL video (es. .mp4).")
            cap = cv2.VideoCapture(src); base_name = name_from_url(src, default="video.mp4")
        else:  # webcam
            idx = int(self.webcam_index_var.get())
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
            # imposta risoluzione ridotta per più FPS su CPU
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cam_w_var.get())
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h_var.get())
            base_name = f"webcam{idx}.mp4"

        if not cap.isOpened():
            return messagebox.showerror("Errore", "Impossibile aprire la sorgente video.")

        self.cap = cap
        self.running = True
        self.video_base_name = base_name
        self.writer = None
        self.last_time = time.time()
        self.frames_count = 0
        self.last_boxes = None
        self.last_scores = None
        self.frame_idx = 0

        self.status.set(f"Streaming… (detector: {self.detector_name})")
        self._video_loop()

    def on_stop_video(self):
        self.running = False
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        if self.writer is not None:
            try: self.writer.release()
            except Exception: pass
            self.writer = None
        self.fps_var.set("FPS: -")

    def _video_loop(self):
        if not self.running or self.cap is None: return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.on_stop_video(); self.status.set("Sorgente terminata."); return

        self.frame_idx += 1
        # ridimensiona per inferenza
        proc_max = int(self.proc_max_side_var.get())
        h, w = frame.shape[:2]
        scale = min(1.0, float(proc_max) / max(h, w)) if proc_max > 0 else 1.0
        f_infer = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame

        # frame skipping: esegui il detector ogni N frame
        N = max(1, int(self.frame_skip_var.get()))
        run_detector = (self.frame_idx % N == 1) or (self.last_boxes is None)

        if run_detector:
            try:
                img_rgb = cv2.cvtColor(f_infer, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img_rgb)
                tensor = self.preprocess(pil).to("cpu")
                with torch.no_grad():
                    out = self.model([tensor])[0]
                boxes  = out["boxes"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                person = (labels == 1)
                boxes = boxes[person]; scores = scores[person]
                if scale < 1.0 and len(boxes) > 0:
                    boxes = boxes / scale  # riporta a risoluzione originale
                self.last_boxes, self.last_scores = boxes, scores
            except Exception as e:
                self.status.set(f"Errore inferenza: {e}")
                self.after(1, self._video_loop); return

        boxes = self.last_boxes if self.last_boxes is not None else np.empty((0,4))
        scores = self.last_scores if self.last_scores is not None else np.empty((0,))

        # filtro threshold + disegno
        thr = float(self.threshold.get())
        keep = scores >= thr
        boxes_f = boxes[keep]; scores_f = scores[keep]

        disp = frame.copy()
        for (x1,y1,x2,y2), s in zip(boxes_f, scores_f):
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(disp, f"person {s:.2f}", (x1, max(y1-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)
        count = len(boxes_f)
        cv2.putText(disp, f"COUNT: {count} (thr={thr:.2f}, skip={N})", (8,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        # FPS
        self.frames_count += 1
        now = time.time(); dt = now - self.last_time
        if dt >= 1.0:
            fps = self.frames_count / dt
            self.fps_var.set(f"FPS: {fps:.1f}")
            self.frames_count = 0; self.last_time = now

        # mostra su canvas
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(image=tk_img); self.canvas.image = tk_img

        # salvataggio video
        if self.save_video_var.get():
            if self.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                os.makedirs("outputs", exist_ok=True)
                out_path = os.path.join("outputs", f"{os.path.splitext(self.video_base_name)[0]}_annotated.mp4")
                self.writer = cv2.VideoWriter(out_path, fourcc,
                                              self.cap.get(cv2.CAP_PROP_FPS) or 25.0,
                                              (disp.shape[1], disp.shape[0]))
                self.status.set(f"Registrazione → {out_path}")
            self.writer.write(disp)

        self.after(1, self._video_loop)

def main():
    app = PeopleCountingApp()
    app.mainloop()

if __name__ == "__main__":
    main()
