# Vaga Fantasma

**Hackathon SECOMP 2026 — Visão Computacional na Logística Universitária**

Detecção de **vagas fantasma** em salas de estudo e bibliotecas: assentos que
o sistema de contagem tradicional marca como ocupados, mas que na prática estão
sendo desperdiçados — reservados por uma mochila cujo dono saiu há muito tempo.

> **Status:** esqueleto inicial. Estrutura, pipeline e dashboard definidos;
> a calibração e o ajuste fino acontecem no período de desenvolvimento.

---

## O problema

Em época de prova, a biblioteca "lota" sem estar cheia. O aluno marca o lugar
com uma mochila e sai por duas horas. Quem chega vê a sala cheia e vai embora;
a gestão vê 100% de ocupação e não tem o que fazer.

Contar pessoas não resolve isso, porque o assento não está ocupado por uma
pessoa — está ocupado por um objeto.

## A proposta

Classificar cada assento em **três** estados, não dois:

| Estado | Condição |
|---|---|
| 🟢 **Livre** | Nenhuma pessoa e nenhum pertence na região do assento |
| 🔴 **Ocupado** | Pessoa presente (ou pertence dentro do período de tolerância) |
| 🟡 **Fantasma** | Pertence presente **sem pessoa** há mais que o limite configurado |

O estado *fantasma* alimenta um alerta automático para a equipe da biblioteca
e libera a vaga na contagem pública.

## Abordagem de Visão Computacional

- **Detecção:** YOLOv8n pré-treinado no COCO, **sem treino nem fine-tuning** —
  todas as classes necessárias (`person`, `backpack`, `handbag`, `laptop`,
  `bottle`, `book`, `cell phone`) já existem no modelo base.
- **Associação detecção → assento:** *Intersection over Smaller* (IoS) em vez
  de IoU. As caixas têm escalas muito diferentes (uma pessoa em pé vs. uma
  garrafa), e o IoU seria baixo nos dois casos.
- **Planta baixa por homografia:** as detecções são projetadas no plano do
  chão (`cv2.getPerspectiveTransform`), gerando uma vista de cima da sala em
  tempo real. A posição dos assentos no mapa **não é colocada à mão** — vem da
  projeção da base de cada ROI.
- **Camada temporal:** o pipeline de referência é *stateless* — responde
  "o que há neste frame?". O problema da vaga fantasma exige responder
  "há quanto tempo isto está assim?", então cada assento tem uma máquina de
  estados com janela deslizante (anti-flicker) e cronômetro de abandono.

O pipeline completo — **captura → processamento → apresentação** — está
descrito em [`docs/arquitetura.md`](docs/arquitetura.md).

---

## Estrutura do projeto

```text
.
├── run.py                  # Ponto de entrada (CLI + sobe o servidor)
├── vision/
│   ├── capture.py          # Fonte de vídeo plugável: webcam | arquivo | RTSP
│   ├── detector.py         # Wrapper do YOLOv8 (classes COCO)
│   ├── seats.py            # ROIs + máquina de estados temporal  <- núcleo
│   ├── floorplan.py        # Homografia: cena → planta baixa
│   └── pipeline.py         # Orquestra captura → detecção → estado → overlay
├── api/
│   └── main.py             # FastAPI: /health, /api/state, /video_feed (MJPEG)
├── web/
│   └── index.html          # Dashboard (HTML/CSS/JS puro, funciona offline)
├── tools/
│   ├── roi_picker.py       # Calibra as ROIs dos assentos com o mouse
│   ├── plan_picker.py      # Calibra a homografia da planta baixa
│   ├── list_cameras.py     # Descobre os índices de câmera disponíveis
│   └── smoke_test.py       # Valida a máquina de estados sem câmera
├── config/
│   ├── *.example.json      # Exemplos versionados
│   ├── rois.json           # Calibração real (ignorada pelo git)
│   └── homography.json     # Calibração da planta (ignorada pelo git)
├── samples/                # Vídeos de contingência para a demo
└── docs/arquitetura.md
```

## Requisitos

- Python 3.10+
- Webcam (ou celular como webcam via Iriun/DroidCam, ou um arquivo de vídeo)

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

O peso `yolov8n.pt` (~6 MB) é baixado automaticamente na primeira execução.

## Execução

**1. Calibrar os assentos** (arraste o mouse sobre cada cadeira, `s` para salvar):

```powershell
python tools/roi_picker.py --source 0
```

**2. Calibrar a planta baixa** (opcional, mas é o painel que mais impressiona) —
clique os 4 cantos de um retângulo real no chão:

```powershell
python tools/plan_picker.py --source 0 --largura 6 --profundidade 4
```

**3. Subir a aplicação:**

```powershell
python run.py
```

Dashboard em **http://127.0.0.1:8000**.

### Opções úteis

```powershell
python run.py --source 1                    # celular como webcam
python run.py --source samples/demo.mp4     # plano B se a câmera falhar
python run.py --ghost-after 10              # tolerância menor, para a demo
```

| Endpoint | Descrição |
|---|---|
| `GET /` | Dashboard |
| `GET /health` | Health check |
| `GET /api/state` | Estado de cada assento em JSON |
| `GET /video_feed` | Stream MJPEG do vídeo anotado |

---

## Limitações conhecidas

- **Oclusão:** pessoas ou objetos atrás de outros podem não ser detectados.
- **Ângulo da câmera:** ROIs são fixas; mover a câmera exige recalibrar.
- **Tolerância é heurística:** o limite em segundos é um parâmetro de operação,
  não uma verdade — precisa ser ajustado por ambiente.
- **Objetos pequenos:** um celular ou um livro isolado tem taxa de detecção
  menor que uma mochila.
- **Iluminação:** contraluz de janela degrada a detecção.

## Trabalhos futuros

- Re-identificação do dono do pertence, para não alertar quando a mesma pessoa
  retorna com frequência.
- Suporte a múltiplas câmeras e agregação por sala/andar.
- Histórico e previsão de ocupação por horário.
- Integração com o app da universidade.

## Créditos

Ferramentas e recursos de terceiros estão listados em [`CREDITS.md`](CREDITS.md).
