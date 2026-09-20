"""Assentos DINAMICOS: detecta as cadeiras em tempo real, sem calibracao.

Diferenca para seats.py (zonas fixas): aqui nao ha ROI marcada a mao. Cada
ciclo o YOLO detecta as cadeiras, e um rastreador leve casa cada deteccao com
um assento ja conhecido pela proximidade do centro. Assim, se a cadeira for
movida, o monitoramento vai junto; se a camera mudar, nada quebra.

O ponto delicado e a oclusao: quando alguem SENTA, o corpo tapa a cadeira e o
YOLO deixa de detecta-la. Por isso cada assento guarda a ultima posicao
conhecida e sobrevive por `seat_ttl_s` segundos sem ser redetectado. Uma pessoa
sobre a ultima posicao mantem o assento vivo e no estado ocupado.
"""

import time
from typing import Dict, List, Optional, Tuple

from .seats import SeatState, intersection_over_smaller


def _center_dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class DynamicSeat:
    """Um assento rastreado. A caixa e atualizada quando a cadeira reaparece."""

    _next_id = 1

    def __init__(self, box: Tuple[float, float, float, float]):
        self.seat_id = f"C{DynamicSeat._next_id}"
        DynamicSeat._next_id += 1

        self.box = box                      # (x1,y1,x2,y2) normalizado
        self.state = SeatState.LIVRE
        self.last_labels: List[str] = []

        now = time.time()
        self._last_chair_seen = now         # ultima vez que a cadeira foi detectada
        self._last_alive = now              # ultima vez com cadeira, pessoa ou objeto
        self._object_since: Optional[float] = None
        self._state_since = now

        # suavizacao: janela curta para nao piscar
        self._person_hist: List[bool] = []
        self._object_hist: List[bool] = []

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    @property
    def roi(self):
        # Compatibilidade com o render/planta, que esperam .roi como em seats.py
        return self.box

    def update_box(self, box):
        # media suave para a caixa nao tremer entre frames
        a = 0.5
        self.box = tuple(a * n + (1 - a) * o for n, o in zip(box, self.box))
        self._last_chair_seen = time.time()

    def observe(self, dets, iou_threshold: float, ghost_after_s: float):
        """Atualiza o estado do assento com as deteccoes do frame atual."""
        person_now = False
        object_now = False
        labels: List[str] = []

        for det in dets:
            if det.is_seat:
                continue
            if intersection_over_smaller((det.x1, det.y1, det.x2, det.y2), self.box) < iou_threshold:
                continue
            if det.is_person:
                person_now = True
            elif det.is_belonging:
                object_now = True
                labels.append(det.label)

        self._person_hist.append(person_now)
        self._object_hist.append(object_now)
        if len(self._person_hist) > 8:
            self._person_hist.pop(0)
            self._object_hist.pop(0)
        self.last_labels = sorted(set(labels))

        has_person = sum(self._person_hist) > len(self._person_hist) / 2
        has_object = sum(self._object_hist) > len(self._object_hist) / 2

        now = time.time()
        if has_person or has_object:
            self._last_alive = now

        if has_person:
            self._object_since = None
            new_state = SeatState.OCUPADO
        elif has_object:
            if self._object_since is None:
                self._object_since = now
            new_state = SeatState.FANTASMA if (now - self._object_since) >= ghost_after_s else SeatState.OCUPADO
        else:
            self._object_since = None
            new_state = SeatState.LIVRE

        if new_state != self.state:
            self.state = new_state
            self._state_since = now

    def is_stale(self, seat_ttl_s: float) -> bool:
        """Descartar so quando a cadeira sumiu E nao ha pessoa nem objeto."""
        gone = time.time() - self._last_chair_seen
        idle = time.time() - self._last_alive
        return gone > seat_ttl_s and idle > seat_ttl_s

    @property
    def seconds_in_state(self) -> float:
        return time.time() - self._state_since

    @property
    def abandoned_for(self) -> float:
        return 0.0 if self._object_since is None else time.time() - self._object_since

    def to_dict(self) -> Dict:
        return {
            "id": self.seat_id,
            "roi": list(self.box),
            "state": self.state.value,
            "labels": self.last_labels,
            "seconds_in_state": round(self.seconds_in_state, 1),
            "abandoned_for": round(self.abandoned_for, 1),
        }


class DynamicSeatManager:
    """Cria, casa, atualiza e descarta assentos a partir das cadeiras detectadas."""

    def __init__(self, match_dist: float = 0.18, seat_ttl_s: float = 8.0, min_seat_conf: float = 0.30):
        self.seats: List[DynamicSeat] = []
        self.match_dist = match_dist        # distancia maxima (norm.) para casar cadeira <-> assento
        self.seat_ttl_s = seat_ttl_s
        self.min_seat_conf = min_seat_conf

    def update(self, detections, iou_threshold: float, ghost_after_s: float):
        chairs = [d for d in detections if d.is_seat and d.conf >= self.min_seat_conf]

        # casa cada cadeira detectada com o assento conhecido mais proximo
        used = set()
        for ch in chairs:
            c = (ch.cx, ch.cy)
            best, best_d = None, self.match_dist
            for seat in self.seats:
                if id(seat) in used:
                    continue
                d = _center_dist(c, seat.center)
                if d < best_d:
                    best, best_d = seat, d
            if best is not None:
                best.update_box((ch.x1, ch.y1, ch.x2, ch.y2))
                used.add(id(best))
            else:
                self.seats.append(DynamicSeat((ch.x1, ch.y1, ch.x2, ch.y2)))

        # estado de cada assento e descarte dos obsoletos
        for seat in self.seats:
            seat.observe(detections, iou_threshold, ghost_after_s)
        self.seats = [s for s in self.seats if not s.is_stale(self.seat_ttl_s)]
