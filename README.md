# Vaga Fantasma

**Hackathon SECOMP 2026 · Visão Computacional na Logística Universitária**

Detecção de **vagas fantasma** em salas de estudo, bibliotecas e salas de aula:
assentos que os sistemas de contagem tradicionais marcam como ocupados, mas que
na prática estão sendo desperdiçados, reservados por uma mochila cujo dono saiu
há muito tempo.

## Equipe

Famosinhos do Algoritmo

- Matheus Lucas Tavares Bueno
- Lucas Assunção Zanon
- Frederico Pires de Moraes Gomes
- Robson Dias Carvalho Soares
- Gustavo Felipe Ferreira Soares

Slides do pitch: [`Equipe_08_Famosinhos_do_Algoritmo.pptx`](Equipe_08_Famosinhos_do_Algoritmo.pptx)

## Objetivo

Dar à universidade uma medida **real** de ocupação dos espaços de estudo. Em vez
de contar apenas pessoas, o sistema distingue quem está de fato usando o assento
de quem apenas o reservou com um objeto. Assim a gestão enxerga a capacidade
verdadeira do ambiente, os alunos sabem se vale a pena ir até lá, e o campus
aproveita melhor o espaço que já tem, sem precisar construir mais.

---

## O problema

Em época de prova, a biblioteca "lota" sem estar cheia. O aluno marca o lugar
com uma mochila e sai por duas horas. Quem chega vê a sala cheia e vai embora, e
a gestão vê 100% de ocupação e conclui que falta espaço.

Contar pessoas não resolve isso, porque o assento não está ocupado por uma
pessoa. Está ocupado por um objeto, e todo sistema de contagem comum é cego para
essa diferença.

## A proposta

Classificar cada assento em **três** estados, não dois:

| Estado | Condição |
|---|---|
| 🟢 **Livre** | Nenhuma pessoa e nenhum pertence na região do assento |
| 🔴 **Ocupado** | Pessoa presente, ou pertence dentro do período de tolerância |
| 🟡 **Fantasma** | Pertence presente **sem pessoa** há mais que o limite configurado |

O estado *fantasma* alimenta um alerta automático para a equipe do espaço e
libera a vaga na contagem pública. O painel ainda contrapõe a **ocupação
aparente** (o que um contador comum veria) à **ocupação real**, tornando visível
exatamente quanta vaga está sendo desperdiçada.

## Abordagem de Visão Computacional

- **Detecção:** YOLOv8 pré-treinado no COCO, **sem treino nem fine-tuning**.
  Todas as classes necessárias (`person`, `chair`, `backpack`, `handbag`,
  `laptop`, `bottle`, `book`, `cell phone`) já existem no modelo base.
- **Assentos dinâmicos (modo usado na demonstração):** no modo `--dynamic`, o
  próprio YOLO detecta as cadeiras a cada ciclo e um rastreador leve as segue.
  Mover a cadeira ou a câmera não quebra nada, e não há passo de calibração.
- **Associação detecção ↔ assento:** *Intersection over Smaller* (IoS) no lugar
  do IoU. As caixas têm escalas muito diferentes (uma pessoa em pé contra uma
  garrafa), e o IoU seria baixo nos dois casos.
- **Camada temporal:** cada assento tem uma máquina de estados com janela
  deslizante (anti-flicker) e cronômetro de abandono. É o que permite responder
  "há quanto tempo isto está assim?", e não só "o que há neste frame?".
- **Planta baixa por homografia (recurso opcional):** as detecções podem ser
  projetadas no plano do chão (`cv2.getPerspectiveTransform`), gerando uma vista
  de cima da sala. Não foi usada na demonstração final, que rodou no modo
  automático, mas fica disponível no projeto.
- **Pipeline em paralelo:** captura, detecção e exibição rodam em threads
  separadas, então o vídeo continua fluido mesmo com a inferência rodando ao
  lado e mesmo sobre uma câmera de rede (celular).

O pipeline completo (captura, processamento e apresentação) está descrito em
[`docs/arquitetura.md`](docs/arquitetura.md).

---

## Estrutura do projeto

```text
.
├── run.py                  # Ponto de entrada (CLI + sobe o servidor)
├── vision/
│   ├── capture.py          # Fonte de vídeo plugável: webcam, arquivo, celular, RTSP
│   ├── detector.py         # Wrapper do YOLOv8 (classes COCO)
│   ├── seats.py            # ROIs fixas + máquina de estados temporal
│   ├── dynamic_seats.py    # Assentos detectados e rastreados em tempo real
│   ├── floorplan.py        # Homografia: cena para planta baixa
│   └── pipeline.py         # Orquestra captura, detecção, estado e overlay
├── api/
│   └── main.py             # FastAPI: /health, /api/state, /video_feed (MJPEG)
├── web/
│   └── index.html          # Dashboard (HTML/CSS/JS puro, funciona offline)
├── tools/
│   ├── roi_picker.py       # Calibra as ROIs dos assentos com o mouse
│   ├── plan_picker.py      # Calibra a homografia da planta baixa
│   ├── list_cameras.py     # Descobre os índices de câmera disponíveis
│   ├── test_stream.py      # Descobre a URL do stream do celular
│   └── smoke_test.py       # Valida a máquina de estados sem câmera
├── config/                 # Calibrações (os *.example.json ficam versionados)
├── samples/                # Vídeos de contingência para a demo
└── docs/arquitetura.md
```

## Requisitos

- Python 3.10 ou superior
- Uma câmera: webcam, celular como câmera IP, ou um arquivo de vídeo

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

O peso `yolov8n.pt` (cerca de 6 MB) é baixado automaticamente na primeira
execução.

## Execução

Há dois modos. O **dinâmico** dispensa calibração e é o mais simples de rodar.

### Modo dinâmico (recomendado), sem calibração

O sistema detecta as cadeiras em tempo real. Se a cadeira for movida, o
monitoramento vai junto, e se a câmera mudar, nada quebra. Basta apontar e rodar:

```powershell
python run.py --source 0 --dynamic --ghost-after 10
```

Ressalva: quando alguém **senta**, o corpo pode tapar a cadeira e o YOLO perdê-la
por instantes. O assento guarda a última posição e sobrevive alguns segundos sem
ser redetectado, então isso não o faz sumir.

### Usando o celular como câmera (ângulo de cima)

Com um app de câmera IP (ex.: IP Camera Lite no iOS), aponte a fonte para a URL
do stream. O ângulo elevado separa melhor as cadeiras:

```powershell
python run.py --source http://IP_DO_CELULAR:8081/ --dynamic --ghost-after 10
```

`python tools/test_stream.py IP_DO_CELULAR` descobre a URL certa do stream.

### Modo com zonas fixas (calibrado)

Mais estável quando a câmera fica parada.

```powershell
python tools/roi_picker.py --source 0                          # marca as cadeiras
python tools/plan_picker.py --source 0 --largura 6 --profundidade 4   # planta (opcional)
python run.py
```

Dashboard em **http://127.0.0.1:8000**.

### Opções úteis

```powershell
python run.py --dynamic                      # cadeiras detectadas em tempo real
python run.py --source samples/demo.mp4      # plano B se a câmera falhar
python run.py --ghost-after 10               # tolerância menor, para a demo
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
- **Ângulo e distância:** cadeiras muito distantes ou muito de lado detectam
  pior. O ângulo de cima ajuda bastante.
- **Tolerância é heurística:** o limite em segundos é um parâmetro de operação,
  não uma verdade, e precisa ser ajustado por ambiente.
- **Objetos pequenos:** um celular ou um livro isolado tem taxa de detecção
  menor que uma mochila.
- **Iluminação:** contraluz de janela degrada a detecção.

## Trabalhos futuros

- Re-identificação do dono do pertence, para não alertar quando a mesma pessoa
  retorna com frequência.
- Suporte a múltiplas câmeras e agregação por sala e por andar.
- Histórico e previsão de ocupação por horário.
- Integração com o app da universidade.

## Créditos

Ferramentas e recursos de terceiros, incluindo o uso de IA generativa, estão
listados em [`CREDITS.md`](CREDITS.md).
