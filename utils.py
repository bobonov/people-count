import cv2, torch, numpy as np
from PIL import Image

COCO_PERSON_CLASS = 1  # 'person' in COCO

def load_model(detector="fast"):
    """
    Carica un detector pre-addestrato su COCO per CPU:
    - detector="fast"     -> SSDLite320_MobileNetV3_Large (molto più veloce su CPU)
    - detector="accurate" -> Faster R-CNN ResNet50 FPN (più preciso ma lento su CPU)
    Ritorna: (model, preprocess, detector_name)
    """
    if detector == "accurate":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        name = "fasterrcnn_resnet50_fpn"
    else:
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = ssdlite320_mobilenet_v3_large(weights=weights)
        name = "ssdlite320_mobilenet_v3_large"

    model.eval().to("cpu")
    preprocess = weights.transforms()
    return model, preprocess, name

def detect_people(model, preprocess, img_bgr, score_thr=0.6):
    """
    Esegue l'inferenza su CPU e ritorna (boxes_filtrate, scores_filtrati).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    tensor = preprocess(img_pil).to("cpu")
    with torch.no_grad():
        outputs = model([tensor])[0]  # i detector vogliono una LISTA di tensori

    boxes  = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    keep = (labels == COCO_PERSON_CLASS) & (scores >= score_thr)
    return boxes[keep], scores[keep]

def draw_boxes_and_count(img_bgr, boxes, scores, score_thr):
    vis = img_bgr.copy()
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"person {s:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    count = len(boxes)
    cv2.putText(vis, f"COUNT: {count} (thr={score_thr})", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return vis, count
