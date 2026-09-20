# Créditos

O regulamento do Hackathon SECOMP 2026 exige que todas as ferramentas,
bibliotecas, APIs, modelos e datasets externos sejam creditados no repositório
e durante a apresentação. Esta é a lista.

## Modelos pré-treinados

| Recurso | Uso no projeto | Licença |
|---|---|---|
| [YOLOv8n](https://github.com/ultralytics/ultralytics) (Ultralytics) | Detecção de pessoas e pertences. Usado **pré-treinado no COCO, sem fine-tuning**. | AGPL-3.0 |
| [COCO dataset](https://cocodataset.org/) | Origem das classes do modelo (`person`, `backpack`, `laptop`, …). Não redistribuímos o dataset. | CC BY 4.0 |

## Bibliotecas

| Biblioteca | Uso |
|---|---|
| [Ultralytics](https://docs.ultralytics.com/) | Carregamento e inferência do YOLO |
| [OpenCV](https://opencv.org/) | Captura de vídeo, desenho do overlay, codificação JPEG |
| [FastAPI](https://fastapi.tiangolo.com/) | Backend HTTP e stream MJPEG |
| [Uvicorn](https://www.uvicorn.org/) | Servidor ASGI |
| [NumPy](https://numpy.org/) | Manipulação de arrays de imagem |

O dashboard é HTML/CSS/JS puro, sem framework nem CDN — decisão tomada para que
funcione sem internet no dia do pitch.

## Material de referência do evento

O projeto-exemplo distribuído pela organização
(`basic-sentiment-analysis`, DeepFace + FastAPI + Streamlit) foi usado como
**referência de arquitetura**: dele reaproveitamos o padrão de separar backend
de inferência e frontend de visualização, e a prática de encapsular erros de
inferência em resposta estruturada em vez de exceção.

Não reaproveitamos código-fonte dele: o domínio (emoção facial, *stateless*) e
o nosso (ocupação de assentos, *stateful*) são diferentes o bastante para que o
pipeline tenha sido escrito do zero. A principal divergência técnica está
documentada em `docs/arquitetura.md`.

## Ferramentas de IA generativa

Conforme o regulamento, o uso de IA generativa está declarado aqui e será
explicitado no pitch.

- **Claude (Anthropic), via Claude Code** — usado em duas frentes:
  1. **Brainstorming e análise de requisitos:** leitura dos PDFs de orientação,
     mapeamento dos critérios de avaliação e comparação de alternativas de
     stack e de tema.
  2. **Geração do esqueleto inicial do repositório:** estrutura de pastas,
     boilerplate do FastAPI/CLI, CSS do dashboard e primeira versão dos
     docstrings.

  As decisões de arquitetura, a modelagem da máquina de estados, a escolha do
  IoS sobre o IoU, os parâmetros de calibração e o ajuste do comportamento em
  campo são da equipe, e todos os integrantes conseguem explicar o
  funcionamento de cada componente.

<!-- TODO(equipe): acrescentar aqui qualquer outra ferramenta usada durante o
     desenvolvimento, e revisar esta seção antes do pitch. -->

## Equipe

<!-- TODO(equipe): preencher com os nomes dos 5 integrantes. -->
