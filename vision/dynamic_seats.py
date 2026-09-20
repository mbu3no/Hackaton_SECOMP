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

    def is_stale(self, seat_ttl_s: float, empty_ttl_s: float) -> bool:
        """Quando descartar o assento.

        Vazio (livre) que perdeu a cadeira: descarta RAPIDO. Sem isso, ao
        arrastar a cadeira, o assento antigo fica pendurado no lugar de origem.
        Com pessoa ou objeto: mantem o prazo longo, para sobreviver a oclusao
        (alguem sentado tapa a cadeira, mas o assento nao pode sumir)."""
        gone = time.time() - self._last_chair_seen
        if self.state == SeatState.LIVRE:
            return gone > empty_ttl_s
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


def _seat_num(seat) -> int:
    """Ordem de criacao a partir do id 'C<n>', para saber qual e o mais antigo."""
    try:
        return int(seat.seat_id[1:])
    except (ValueError, IndexError):
        return 0


class DynamicSeatManager:
    """Cria, casa, atualiza e descarta assentos a partir das cadeiras detectadas.

    O YOLO frequentemente devolve varias caixas para a MESMA cadeira (a caixa
    treme entre frames, ou a cadeira e detectada em duas classes). Sem cuidado,
    cada caixa vira um assento e uma cadeira e contada varias vezes. Por isso ha
    tres defesas: deduplicar as deteccoes do frame, casar por sobreposicao (nao
    so por distancia) e, ao final, mesclar assentos que se sobrepoem.
    """

    def __init__(self, match_dist: float = 0.22, seat_ttl_s: float = 8.0,
                 empty_ttl_s: float = 1.5, min_seat_conf: float = 0.25,
                 dedup_ios: float = 0.5, merge_ios: float = 0.5, max_area: float = 0.45):
        self.seats: List[DynamicSeat] = []
        self.match_dist = match_dist        # distancia (norm.) para casar por centro
        self.seat_ttl_s = seat_ttl_s        # sobrevivencia com pessoa/objeto (oclusao)
        self.empty_ttl_s = empty_ttl_s      # sobrevivencia de assento vazio sem cadeira
        self.min_seat_conf = min_seat_conf
        self.dedup_ios = dedup_ios          # sobreposicao acima da qual duas deteccoes sao a mesma
        self.merge_ios = merge_ios          # sobreposicao acima da qual dois assentos sao fundidos
        self.max_area = max_area            # caixa maior que isto (fracao do frame) e descartada

    @staticmethod
    def _area(box) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    def _dedup(self, chairs):
        """NMS simples: mantem as cadeiras de maior confianca e descarta as que
        se sobrepoem demais a uma ja mantida (a mesma cadeira detectada 2x)."""
        kept = []
        for ch in sorted(chairs, key=lambda d: d.conf, reverse=True):
            box = (ch.x1, ch.y1, ch.x2, ch.y2)
            if self._area(box) > self.max_area:
                continue  # caixa gigante (englobaria varias cadeiras)
            if any(intersection_over_smaller(box, k) >= self.dedup_ios for k in kept):
                continue
            kept.append(box)
        return kept

    def _match(self, box):
        """Assento existente que melhor corresponde a esta caixa, ou None.

        Prefere sobreposicao (robusto a cadeira que balanca) e, na falta,
        proximidade de centro (robusto a cadeira arrastada devagar)."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best, best_score = None, 0.0
        for seat in self.seats:
            score = intersection_over_smaller(box, seat.box)
            if score >= 0.3 and score > best_score:
                best, best_score = seat, score
        if best is not None:
            return best
        # sem sobreposicao: tenta pelo centro mais proximo
        best, best_d = None, self.match_dist
        for seat in self.seats:
            d = _center_dist((cx, cy), seat.center)
            if d < best_d:
                best, best_d = seat, d
        return best

    def _merge_overlapping(self):
        """Funde assentos sobrepostos, mantendo sempre o mais antigo (que ja
        acumulou estado e cronometro)."""
        kept: List[DynamicSeat] = []
        for seat in sorted(self.seats, key=_seat_num):   # mais antigo primeiro
            if any(intersection_over_smaller(seat.box, k.box) >= self.merge_ios for k in kept):
                continue
            kept.append(seat)
        self.seats = kept

    def update(self, detections, iou_threshold: float, ghost_after_s: float):
        chairs = [d for d in detections if d.is_seat and d.conf >= self.min_seat_conf]
        boxes = self._dedup(chairs)

        # casa cada caixa (ja deduplicada) com um assento existente, ou cria um
        used = set()
        for box in boxes:
            seat = self._match(box)
            if seat is not None and id(seat) not in used:
                seat.update_box(box)
                used.add(id(seat))
            elif seat is None:
                self.seats.append(DynamicSeat(box))

        # estado de cada assento, mescla de duplicados e descarte dos obsoletos
        for seat in self.seats:
            seat.observe(detections, iou_threshold, ghost_after_s)
        self._merge_overlapping()
        self.seats = [s for s in self.seats
                      if not s.is_stale(self.seat_ttl_s, self.empty_ttl_s)]
