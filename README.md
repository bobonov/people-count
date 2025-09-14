# People Counting (solo inferenza, CPU-friendly)

Conta quante persone sono presenti in un'immagine (o in un video) usando un detector pre-addestrato
**Faster R-CNN ResNet50 FPN** (COCO). Nessun training, solo inferenza.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

se si utilizza Linux/Mac tkinter va installato tramite il gestore di pacchetti
