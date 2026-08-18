# rag-pdf-local

Pipeline RAG (Retrieval-Augmented Generation) **100% local**: extrai texto de um PDF, indexa em
um banco vetorial local (Chroma) e responde perguntas usando um LLM servido pelo **LM Studio**
(API local compatível com OpenAI). Nenhum dado sai da máquina.

## Requisitos

- Python >= 3.11
- [LM Studio](https://lmstudio.ai/) instalado, com **três** modelos carregados e o servidor local
  ativo (aba **Developer** → **Start Server**, expõe `http://localhost:1234/v1` por padrão):
  um modelo de chat (ex.: `llama-3.3-8b-instruct-128k`), um modelo de embedding
  (`EmbeddingGemma`) e, para a aba de imagens, um modelo **com suporte a visão** (por padrão,
  `gemma4-12b-qat-uncensored-hauhaucs-balanced`). Confira os ids exatos com
  `curl http://localhost:1234/v1/models` e ajuste
  `LM_STUDIO_MODEL`/`LM_STUDIO_EMBEDDING_MODEL`/`LM_STUDIO_VISION_MODEL` no `.env`.
  - Prefira modelos **sem "thinking"** (raciocínio interno antes da resposta), tanto o de chat
    quanto o de visão. Modelos "thinking" ficam mais lentos por chamada — o que pesa bastante no
    resumo map-reduce, que faz várias chamadas — e alguns vazam o raciocínio dentro da própria
    resposta quando essa extração é desativada nas configs do LM Studio. No modelo de visão, o
    raciocínio ainda consome o orçamento de `VISION_MAX_TOKENS` antes de escrever a descrição —
    se a descrição de algumas imagens vier cortada, aumente `VISION_MAX_TOKENS` no `.env` (o
    cliente detecta esse corte e levanta um erro claro em vez de salvar uma descrição incompleta).
- Para a aba de imagens: o embedding visual **SigLIP** (`cstr/siglip-so400m-patch14-384-GGUF`,
  1152 dimensões) roda via **CUDA nativo**, fora do LM Studio (que ainda não suporta esse
  modelo), através do [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).
  - Baixe `siglip-so400m-patch14-384.gguf` e salve em `models/siglip-so400m-patch14-384.gguf`.
  - Builde e suba o servidor local com os scripts em
    [embedding_image_server/](embedding_image_server/) (`setup.ps1` builda com CUDA,
    `start-server.ps1` sobe o servidor) — veja
    [embedding_image_server/README.md](embedding_image_server/README.md) para os pré-requisitos
    (CMake, Visual Studio Build Tools, CUDA Toolkit).
  - Ajuste `CRISPEMBED_BASE_URL` no `.env` se usar outra porta.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
make install          # pip install -e ".[dev]" + pre-commit install

cp .env.example .env  # ajuste os modelos/paths conforme necessário
```

## Uso

### Script de conveniência (recomendado no dia a dia)

**Linux/macOS** (bash):

```bash
scripts/rag-pdf.sh start     # ou: make up
scripts/rag-pdf.sh status    # ou: make status
scripts/rag-pdf.sh restart   # ou: make restart
scripts/rag-pdf.sh stop      # ou: make down
scripts/rag-pdf.sh clean     # ou: make clean-db  — apaga todos os documentos indexados (pede confirmação)
scripts/rag-pdf.sh logs      # ou: make logs      — acompanha o log da interface web
```

**Windows** (PowerShell nativo — não precisa de bash/Git Bash):

```powershell
.\scripts\rag-pdf.ps1 start
.\scripts\rag-pdf.ps1 status
.\scripts\rag-pdf.ps1 restart
.\scripts\rag-pdf.ps1 stop
.\scripts\rag-pdf.ps1 clean   # apaga todos os documentos indexados (pede confirmação)
.\scripts\rag-pdf.ps1 logs    # acompanha o log da interface web
```

> Chamar `scripts\rag-pdf.sh` (o script bash) direto de um PowerShell não funciona bem: a
> associação de arquivo `.sh` do Windows abre uma janela separada do Git Bash sem mostrar nada
> no terminal atual, e mesmo prefixando com `bash`, o bash interpreta `\` como escape (não como
> separador de caminho) — `.\scripts\rag-pdf.sh` vira um caminho quebrado. Por isso o
> `rag-pdf.ps1` existe: mesmo conjunto de comandos, 100% PowerShell nativo.

`start` já confere se o LM Studio (e o CrispEmbed, se configurado) está respondendo antes de
subir a interface, e `status` mostra se está rodando, essas conexões e os documentos indexados
no vector store.

### CLI

```bash
# 1. Indexar um PDF
rag-pdf ingest --pdf data/raw/documento.pdf

# 2. Perguntar (busca top-k fixa)
rag-pdf ask "Qual o prazo de garantia descrito no documento?"

# 3. Resumir o documento inteiro (map-reduce, ignora perguntas pontuais)
rag-pdf summarize --source documento.pdf
```

### Interface web (Streamlit)

```bash
pip install -e ".[app]"   # se ainda não tiver instalado
make app                  # ou: streamlit run src/rag_pdf/app/streamlit_app.py
                           # ou: scripts/rag-pdf.sh start (Linux/macOS, sobe em background)
                           # ou: .\scripts\rag-pdf.ps1 start (Windows, sobe em background)
```

Abre em `http://localhost:8501`, com:

- **Painel central "🗂️ Pastas e arquivos"** (colapsado por padrão): árvore única que
  unifica seleção e gerenciamento — checkbox por arquivo/pasta pra restringir a busca (sem
  seleção = busca em tudo), e ícones de ação: 🔀 mover (popover), ✏️ renomear pasta
  (popover), ➕ criar subpasta/pasta vazia (popover), 🗑️ apagar (confirmação em duas etapas)
- **Barra lateral**: upload/indexação de PDF (escolhendo a pasta virtual pela árvore),
  contador de chunks, e o seletor de **modo de resposta**:
  - **Top-k fixo**: sempre busca os `k` trechos mais parecidos com a pergunta — rápido, bom
    para fatos pontuais
  - **Busca adaptativa**: pega o trecho mais parecido e vai incluindo os próximos enquanto
    estiverem dentro de uma margem de similaridade, até um teto de contexto — 3 perfis
    prontos (Restrita 30%/3k chars, Padrão 45%/6k chars, Abrangente 60%/14k chars) ou
    "Personalizado" com sliders manuais
  - **Resumo (map-reduce)**: lê o documento inteiro em blocos e resume — ignora o texto da
    pergunta, bom para "quais os principais pontos?"; mostra barra de progresso com tempo
    estimado
- **Chat**: histórico de perguntas/respostas com as fontes usadas (arquivo, página,
  distância)

### Aba "🖼️ Imagens"

- **Indexar imagens**: sobe uma ou mais imagens (PNG/JPG/WEBP) — para cada uma, um modelo com
  visão no LM Studio gera uma descrição textual completa, que é embutida com o mesmo
  EmbeddingGemma dos PDFs (768d); a imagem em si é embutida com SigLIP via CrispEmbed (1152d).
  Os dois embeddings, a descrição e o caminho da imagem são salvos no vector store.
- **Buscar imagens**: seletor com dois modos —
  - **Por texto**: descreve o que procura, compara com os embeddings de descrição.
  - **Por imagem parecida**: sobe uma imagem de exemplo, compara com os embeddings visuais
    (SigLIP) das imagens já indexadas — útil quando é mais fácil mostrar do que descrever.
- **Galeria**: miniaturas de todas as imagens indexadas, com exclusão individual.

Veja [docs/PIPELINE.md](docs/PIPELINE.md) para uma explicação detalhada, arquivo por arquivo,
de como cada etapa funciona.

## Desenvolvimento

```bash
make test    # pytest + cobertura
make lint    # ruff + mypy
make format  # ruff format
```

## Estrutura

```
src/rag_pdf/
├── config.py           # settings via pydantic-settings (.env)
├── ingestion/           # extração (pypdf) e limpeza do texto do PDF
├── processing/           # chunking, embeddings de texto (Gemma) e de imagem (SigLIP/CrispEmbed)
├── vectorstore/          # Chroma (vector store local, embutido) — coleções de PDF e de imagem
├── llm/                  # clientes HTTP para o LM Studio: chat/QA e descrição de imagem (visão)
├── retrieval/             # busca top-k fixa e busca adaptativa (margem + teto de contexto)
├── prompts/                # templates de prompt do QA, resumo (map-reduce) e descrição de imagem
├── pipeline/                # orquestração: ingest/query de PDF e ingest/query de imagem
├── cli/                      # interface de linha de comando (Typer)
└── app/                       # interface web (Streamlit), com abas de PDFs e Imagens

scripts/
├── rag-pdf.sh                  # start/stop/restart/status/clean/logs (Linux/macOS, bash)
└── rag-pdf.ps1                 # o mesmo, nativo em PowerShell (Windows)

models/                          # .gguf baixados manualmente (ex.: siglip-so400m-patch14-384.gguf)

embedding_image_server/          # build local do CrispEmbed (CUDA) + scripts de setup/start
├── README.md                     # pré-requisitos e instruções
├── setup.ps1                     # clona e builda o CrispEmbed com CUDA
└── start-server.ps1              # sobe o crispembed-server local
```

Explicação detalhada de cada arquivo, com analogias: [docs/PIPELINE.md](docs/PIPELINE.md).

## Escolhas de custo-benefício

| Componente | Escolha |
|---|---|
| Extração de PDF | `pypdf` (fallback `pdfplumber` para tabelas) |
| Embeddings | `EmbeddingGemma`, servido pelo LM Studio via `/v1/embeddings` |
| Vector store | `Chroma`, embutido, sem servidor |
| LLM | LM Studio (local), consumido via SDK `openai` apontando para `http://localhost:1234/v1` |
| Busca | top-k fixo ou adaptativa (margem de similaridade + teto de contexto), configurável na UI |
| Resumo de documento inteiro | map-reduce em blocos, com barra de progresso e ETA |
| Descrição de imagem | modelo com visão, servido pelo LM Studio |
| Embedding de imagem | `SigLIP` GGUF, servido localmente via CUDA pelo CrispEmbed (LM Studio não suporta ainda) |
