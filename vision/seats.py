"""Maquina de estados temporal por assento.

Este e o nucleo do projeto e o que o pipeline de referencia (stateless) nao faz:
um frame isolado responde "o que existe aqui?", mas o problema da vaga fantasma
exige responder "ha quanto tempo isto esta assim?".

Estados:
    LIVRE     - nada no assento
    OCUPADO   - pessoa presente (ou pertence dentro do periodo de tolerancia)
    FANTASMA  - pertence sem pessoa ha mais que `ghost_after_s`
"""

import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Tuple


class SeatState(str, Enum):
    LIVRE = "livre"
    OCUPADO = "ocupado"
    FANTASMA = "fantasma"


def intersection_over_smaller(a: Tuple[float, float, float, float],
                              b: Tuple[float, float, float, float]) -> float:
    """Area de intersecao dividida pela area da MENOR das duas caixas.

    Escolhido no lugar do IoU porque as caixas tem escalas muito diferentes:
    uma pessoa em pe ocupa uma caixa enorme perto da ROI de um assento, e uma
    garrafa ocupa uma caixa minuscula dentro dela. O IoU seria baixo nos dois
    casos; o IoS captura corretamente "esta dentro".
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


@dataclass
class Seat:
    """Um assento monitorado, com ROI normalizada [0..1] e memoria temporal."""

    seat_id: str
    roi: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalizado

    # Janela deslizante de observacoes cruas, para nao piscar a cada frame ruim.
    _person_hist: Deque[bool] = field(default_factory=lambda: deque(maxlen=10), repr=False)
    _object_hist: Deque[bool] = field(default_factory=lambda: deque(maxlen=10), repr=False)

    state: SeatState = SeatState.LIVRE
    _object_since: float | None = field(default=None, repr=False)
    _state_since: float = field(default_factory=time.time, repr=False)
    last_labels: List[str] = field(default_factory=list)

    def observe(self, detections, iou_threshold: float, ghost_after_s: float) -> SeatState:
        """Atualiza o estado do assento com as deteccoes do frame atual."""
        person_now = False
        object_now = False
        labels: List[str] = []

        for det in detections:
            box = (det.x1, det.y1, det.x2, det.y2)
            if intersection_over_smaller(box, self.roi) < iou_threshold:
                continue
            if det.is_person:
                person_now = True
            elif det.is_belonging:
                object_now = True
                labels.append(det.label)

        self._person_hist.append(person_now)
        self._object_hist.append(object_now)
        self.last_labels = sorted(set(labels))

        # Voto majoritario na janela: absorve deteccao perdida em 1 ou 2 frames.
        has_person = sum(self._person_hist) > len(self._person_hist) / 2
        has_object = sum(self._object_hist) > len(self._object_hist) / 2

        now = time.time()

        if has_person:
            self._object_since = None
            new_state = SeatState.OCUPADO
        elif has_object:
            if self._object_since is None:
                self._object_since = now
            abandoned_for = now - self._object_since
            # Dentro da tolerancia ainda conta como ocupado: a pessoa pode ter
            # ido ao banheiro. Passou disso, a vaga esta sendo desperdicada.
            new_state = SeatState.FANTASMA if abandoned_for >= ghost_after_s else SeatState.OCUPADO
        else:
            self._object_since = None
            new_state = SeatState.LIVRE

        if new_state != self.state:
            self.state = new_state
            self._state_since = now

        return self.state

    @property
    def seconds_in_state(self) -> float:
        return time.time() - self._state_since

    @property
    def abandoned_for(self) -> float:
        return 0.0 if self._object_since is None else time.time() - self._object_since

    def to_dict(self) -> Dict:
        return {
            "id": self.seat_id,
            "roi": list(self.roi),
            "state": self.state.value,
            "labels": self.last_labels,
            "seconds_in_state": round(self.seconds_in_state, 1),
            "abandoned_for": round(self.abandoned_for, 1),
        }


def load_seats(path: str) -> List[Seat]:
    """Le as ROIs calibradas por tools/roi_picker.py."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [Seat(seat_id=s["id"], roi=tuple(s["roi"])) for s in data["seats"]]
