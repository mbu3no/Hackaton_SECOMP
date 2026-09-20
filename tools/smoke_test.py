#!/usr/bin/env python3
"""Teste de fumaca da maquina de estados, sem camera e sem YOLO.

Valida o cenario de uso completo do problema antes de depender de hardware:
livre -> ocupado -> tolerancia -> fantasma -> dono volta -> livre.

Uso:
    python tools/smoke_test.py
"""

import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.seats import Seat, SeatState, intersection_over_smaller  # noqa: E402

ROI = (0.06, 0.38, 0.27, 0.86)
PESSOA = (0.05, 0.10, 0.30, 0.95)   # caixa alta, engloba a ROI
MOCHILA = (0.15, 0.55, 0.19, 0.65)  # caixa pequena, dentro da ROI
FORA = (0.60, 0.10, 0.70, 0.20)
GHOST_AFTER = 1.0
IOS_MIN = 0.25


def det(is_person, box):
    return types.SimpleNamespace(
        is_person=is_person, is_belonging=not is_person, label="mochila",
        x1=box[0], y1=box[1], x2=box[2], y2=box[3],
    )


def main():
    falhas = []

    def checa(nome, obtido, esperado):
        ok = obtido == esperado
        print(f"  {'OK  ' if ok else 'FALHA'}  {nome:<26} -> {obtido}")
        if not ok:
            falhas.append(f"{nome}: esperado {esperado}, obtido {obtido}")

    print("\n[1] Associacao deteccao <-> ROI (IoS)")
    checa("pessoa em pe na ROI", intersection_over_smaller(PESSOA, ROI) >= IOS_MIN, True)
    checa("mochila dentro da ROI", intersection_over_smaller(MOCHILA, ROI) >= IOS_MIN, True)
    checa("objeto fora da ROI", intersection_over_smaller(FORA, ROI) >= IOS_MIN, False)

    print("\n[2] Maquina de estados (ciclo completo)")
    seat = Seat("A1", ROI)

    def alimenta(dets, n=12):
        for _ in range(n):
            seat.observe(dets, IOS_MIN, GHOST_AFTER)

    alimenta([])
    checa("sala vazia", seat.state, SeatState.LIVRE)

    alimenta([det(True, PESSOA)])
    checa("pessoa sentada", seat.state, SeatState.OCUPADO)

    alimenta([det(False, MOCHILA)])
    checa("saiu, dentro da tolerancia", seat.state, SeatState.OCUPADO)

    time.sleep(GHOST_AFTER)
    alimenta([det(False, MOCHILA)])
    checa("passou a tolerancia", seat.state, SeatState.FANTASMA)

    alimenta([det(True, PESSOA), det(False, MOCHILA)])
    checa("dono voltou", seat.state, SeatState.OCUPADO)
    checa("cronometro zerado", seat.abandoned_for < 0.5, True)

    alimenta([])
    checa("levou tudo embora", seat.state, SeatState.LIVRE)

    print("\n[3] Anti-flicker (deteccao perdida em 1 frame nao muda o estado)")
    seat2 = Seat("A2", ROI)
    for _ in range(12):
        seat2.observe([det(True, PESSOA)], IOS_MIN, GHOST_AFTER)
    seat2.observe([], IOS_MIN, GHOST_AFTER)  # 1 frame ruim
    checa("1 frame perdido", seat2.state, SeatState.OCUPADO)

    print()
    if falhas:
        print(f"{len(falhas)} FALHA(S):")
        for f in falhas:
            print(f"  - {f}")
        return 1
    print("Todos os testes passaram.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
