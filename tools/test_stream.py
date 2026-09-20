#!/usr/bin/env python3
"""Descobre a URL certa do stream do celular e testa se o PC consegue ler.

Apps de camera IP servem o MJPEG em caminhos diferentes (/live, /video,
/videofeed...). Em vez de adivinhar, este script testa todos.

Muitos desses apps exigem usuario e senha (HTTP Basic). Nesse caso passe
as credenciais, que sao embutidas na URL no formato usuario:senha@host.

Uso:
    python tools/test_stream.py 192.168.0.17
    python tools/test_stream.py 192.168.0.17:8081
    python tools/test_stream.py 192.168.0.17:8081 admin senha123
    python tools/test_stream.py http://admin:senha123@192.168.0.17:8081/live
"""

import os
import sys
import time

import cv2

CAMINHOS = ["/live", "/video", "/videofeed", "/mjpeg", "/stream", "/cam.mjpg", "/"]
PORTAS = [8081, 8080, 81, 8000]


def testa(url: str, timeout_s: float = 6.0):
    """Tenta abrir a URL e ler um frame. Devolve (ok, largura, altura)."""
    cap = cv2.VideoCapture(url)
    inicio = time.time()
    try:
        while time.time() - inicio < timeout_s:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    return True, w, h
            time.sleep(0.2)
        return False, 0, 0
    finally:
        cap.release()


def candidatas(arg: str, cred: str = ""):
    """Gera as URLs a testar. `cred` e "usuario:senha@" ou string vazia."""
    if arg.startswith("http"):
        yield arg
        return
    host = arg.split("/")[0]
    portas = [host.split(":")[1]] if ":" in host else PORTAS
    host = host.split(":")[0]
    for porta in portas:
        for c in CAMINHOS:
            yield f"http://{cred}{host}:{porta}{c}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    # Silencia o ruido do backend FFMPEG nas tentativas que falham.
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

    print("Testando URLs. A primeira que entregar um frame e a certa.\n")
    for url in candidatas(sys.argv[1]):
        print(f"  {url:<44}", end="", flush=True)
        ok, w, h = testa(url, timeout_s=4.0)
        if ok:
            print(f"OK  {w}x{h}")
            print(f"\nFUNCIONOU. Use esta URL:\n")
            print(f"  python tools/roi_picker.py  --source {url}")
            print(f"  python tools/plan_picker.py --source {url}")
            print(f"  python run.py --source {url} --ghost-after 10\n")
            return 0
        print("falhou")

    print("\nNenhuma URL respondeu. Verifique:")
    print("  1. O celular e o PC estao na MESMA rede WiFi?")
    print("  2. O servidor esta ligado no app (botao de iniciar)?")
    print("  3. O IP digitado e o que o app mostra na tela?")
    print("  4. A rede pode estar isolando dispositivos (comum em WiFi de evento).")
    print("     Nesse caso, use o roteador do celular: ligue o hotspot do iPhone")
    print("     e conecte o PC nele. Ai os dois ficam na mesma rede.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
