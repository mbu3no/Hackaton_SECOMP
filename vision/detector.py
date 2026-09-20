"""Wrapper do YOLOv8 pre-treinado (COCO).

Nao ha treino nem fine-tuning: todas as classes que o problema precisa
(pessoa e pertences) ja existem no COCO, o que cabe na janela do hackathon.
"""

from dataclasses import dataclass
from typing import List

from ultralytics import YOLO

# IDs das classes COCO usadas pelo projeto.
PERSON_CLASS = 0

# Assentos detectados dinamicamente (modo --dynamic, sem calibracao).
# "sofa" (57) foi removido de proposito: gerava caixas enormes que englobavam
# varias cadeiras e poluiam a contagem.
SEAT_CLASSES = {
    56: "cadeira",
    13: "banco",
}

# Pertences que caracterizam um assento "marcado" por um objeto.
BELONGING_CLASSES = {
    24: "mochila",
    26: "bolsa",
    28: "mala",
    39: "garrafa",
    41: "copo",
    63: "notebook",
    67: "celular",
    73: "livro",
}

KEEP_CLASSES = [PERSON_CLASS] + list(SEAT_CLASSES) + list(BELONGING_CLASSES)


@dataclass
class Detection:
    """Uma deteccao em coordenadas normalizadas [0..1] (independe da resolucao)."""

    cls_id: int
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def is_person(self) -> bool:
        return self.cls_id == PERSON_CLASS

    @property
    def is_belonging(self) -> bool:
        return self.cls_id in BELONGING_CLASSES

    @property
    def is_seat(self) -> bool:
        return self.cls_id in SEAT_CLASSES

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


class Detector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35, imgsz: int = 640):
        # Na primeira execucao o Ultralytics baixa o peso (~6 MB) automaticamente.
        self.model = YOLO(model_path)
        self.conf = conf
        # Lado maior da imagem na entrada da rede. Reduzir acelera bastante em
        # CPU, ao custo de perder objetos pequenos e distantes.
        self.imgsz = imgsz

    def detect(self, frame) -> List[Detection]:
        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            conf=self.conf,
            classes=KEEP_CLASSES,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections: List[Detection] = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if cls_id == PERSON_CLASS:
                label = "pessoa"
            elif cls_id in SEAT_CLASSES:
                label = SEAT_CLASSES[cls_id]
            else:
                label = BELONGING_CLASSES.get(cls_id, "?")
            detections.append(
                Detection(
                    cls_id=cls_id,
                    label=label,
                    conf=float(box.conf[0]),
                    x1=x1 / w,
                    y1=y1 / h,
                    x2=x2 / w,
                    y2=y2 / h,
                )
            )
        return detections
