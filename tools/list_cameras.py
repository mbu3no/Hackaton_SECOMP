#!/usr/bin/env python3
"""Descobre quais indices de camera existem nesta maquina.

Util para achar em que indice o celular apareceu depois de conectar
(Iriun / DroidCam costumam entrar no indice 1 ou 2, depois da webcam interna).

Uso:
    python tools/list_cameras.py
"""

import sys

import cv2


def main():
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    encontradas = []

    print("Procurando cameras nos indices 0..5...\n")
    for idx in range(6):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                print(f"  [{idx}]  OK  -  {w}x{h}")
                encontradas.append(idx)
            else:
                print(f"  [{idx}]  abriu, mas nao entregou frame (em uso por outro app?)")
        cap.release()

    print()
    if not encontradas:
        print("Nenhuma camera encontrada.")
        print("Se o celular deveria aparecer, confira se o app esta conectado")
        print("e se nenhum outro programa esta segurando a camera.")
    else:
        print(f"Cameras disponiveis: {encontradas}")
        print("\nNotebooks costumam expor a camera IR do Windows Hello")
        print("como um indice extra. Se a imagem sair preto e branco ou")
        print("estourada, tente o proximo indice da lista.\n")
        for idx in encontradas:
            print(f"  indice {idx}:")
            print(f"    python tools/roi_picker.py --source {idx}")
            print(f"    python run.py --source {idx}")


if __name__ == "__main__":
    main()
