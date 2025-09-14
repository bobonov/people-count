# People Count — Quick Start

Conta persone in **immagini**, **video** o **webcam** con due detector:
- `fast` → SSDLite (veloce)
- `accurate` → Faster R-CNN (più preciso)
- In GUI puoi anche scegliere `both` per confrontarli sulla stessa immagine.

---

## 1) Setup

```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Se si utilizza Linux/Mac tkinter va installato tramite il gestore di pacchetti

```powershell
# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) GUI — `detect_gui.py`

```bash
python detect_gui.py
```

- **Immagine (File/URL)**: seleziona `fast`, `accurate` o **`both`** per vedere i due risultati affiancati con **tempo di inferenza** (ms).
- **Video**: elabora e salva un MP4 annotato in `outputs/`.
- **Webcam**: avvia/stop; overlay con conteggio e tempo per frame.

## 3) CLI guidata — `detect_cli.py`

```bash
# completamente guidata
python detect_cli.py

# guidata parziale (specifica solo la modalità)
python detect_cli.py --mode video

# non interattiva
python detect_cli.py --mode image --detector accurate --input ./imgs --out outputs --score_thr 0.5
```

Modalità: `image | video | webcam`.

---

## 4) Script da riga di comando

### Immagini
```bash
# singola immagine
python detect_image.py --input ./imgs/foto.jpg --detector fast

# cartella di immagini
python detect_image.py --input ./imgs --detector accurate

# (se supportato) URL immagine
python detect_image.py --input "https://esempio.it/foto.jpg" --detector fast
```

### Video
```bash
python detect_video.py --input ./video.mp4 --detector fast --out outputs --score_thr 0.6
```

### Webcam
```bash
python detect_interactive.py --detector accurate --score_thr 0.6
```

Opzioni comuni:
- `--detector fast|accurate`
- `--score_thr 0.1..0.95` (default 0.6)
- `--out outputs`