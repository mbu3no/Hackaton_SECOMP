"""Projecao da cena para uma planta baixa, via homografia.

A camera ve o chao em perspectiva: um retangulo real vira um trapezio na
imagem. A homografia e a matriz 3x3 que desfaz essa distorcao, permitindo
desenhar a sala vista de cima.

Ponto importante: a homografia so e valida para o PLANO DO CHAO. Projetar a
caixa inteira de um objeto daria erro, porque cadeira e pessoa tem altura.
Por isso projetamos sempre a base da caixa (onde o objeto toca o chao).
"""

import json
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Destino no plano normalizado [0..1]. A ordem espelha a do plan_picker:
# frente-esquerda, frente-direita, fundo-direita, fundo-esquerda.
# "Frente" e o lado proximo da camera, que vai para a base da planta.
DST_CORNERS = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)


class FloorPlan:
    def __init__(self, image_points: List[Tuple[float, float]], size_m: Tuple[float, float]):
        if len(image_points) != 4:
            raise ValueError("A homografia exige exatamente 4 pontos.")
        src = np.array(image_points, dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(src, DST_CORNERS)
        self.size_m = size_m
        self.image_points = image_points

    @classmethod
    def load(cls, path: str) -> Optional["FloorPlan"]:
        """Carrega a calibracao. Devolve None se ainda nao foi feita.

        A ausencia de planta nao pode derrubar a aplicacao: o dashboard
        simplesmente esconde o painel e o resto continua funcionando.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return cls(
                image_points=[tuple(p) for p in data["image_points"]],
                size_m=tuple(data.get("plan_size_m", [6.0, 4.0])),
            )
        except (FileNotFoundError, KeyError, ValueError):
            return None

    def project(self, x: float, y: float) -> Tuple[float, float]:
        """Ponto da imagem (normalizado) -> ponto da planta (normalizado)."""
        pt = np.array([[[x, y]]], dtype=np.float32)
        px, py = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(px), float(py)

    @staticmethod
    def ground_anchor(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Base da caixa: o ponto em que o objeto encosta no chao."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, y2)

    def project_box(self, box) -> Tuple[float, float]:
        return self.project(*self.ground_anchor(box))

    def to_dict(self) -> dict:
        return {"size_m": list(self.size_m), "image_points": [list(p) for p in self.image_points]}
