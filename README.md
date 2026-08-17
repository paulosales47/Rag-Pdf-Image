# rag-pdf-local

Pipeline RAG (Retrieval-Augmented Generation) **100% local**: extrai texto de um PDF, indexa em
um banco vetorial local (Chroma) e responde perguntas usando um LLM servido pelo **LM Studio**
(API local compatível com OpenAI). Nenhum dado sai da máquina.

## Requisitos

- Python >= 3.11
- [LM Studio](https://lmstudio.ai/) instalado, com **dois** modelos carregados e o servidor local
  ativo (aba **Developer** → **Start Server**, expõe `http://localhost:1234/v1` por padrão):
  um modelo de chat (ex.: `llama-3.3-8b-instruct-128k`) e um modelo de embedding
  (`EmbeddingGemma`). Confira os ids exatos com `curl http://localhost:1234/v1/models` e ajuste
  `LM_STUDIO_MODEL`/`LM_STUDIO_EMBEDDING_MODEL` no `.env`.
  - Prefira um modelo **sem "thinking"** (raciocínio interno antes da resposta). Modelos
    "thinking" ficam mais lentos por chamada — o que pesa bastante no resumo map-reduce, que
    faz várias chamadas — e alguns vazam o raciocínio dentro da própria resposta quando essa
    extração é desativada nas configs do LM Studio.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
make install          # pip install -e ".[dev]" + pre-commit install

cp .env.example .env  # ajuste os modelos/paths conforme necessário
```

## Uso

### Script de conveniência (recomendado no dia a dia)

```bash
scripts/rag-pdf.sh start     # ou: make up
scripts/rag-pdf.sh status    # ou: make status
scripts/rag-pdf.sh restart   # ou: make restart
scripts/rag-pdf.sh stop      # ou: make down
scripts/rag-pdf.sh clean     # ou: make clean-db  — apaga todos os documentos indexados (pede confirmação)
scripts/rag-pdf.sh logs      # ou: make logs      — acompanha o log da interface web
```

`start` já confere se o LM Studio está respondendo antes de subir a interface, e `status`
mostra se está rodando, a conexão com o LM Studio e os documentos indexados no vector store.

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
                           # ou: scripts/rag-pdf.sh start (sobe em background)
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
├── processing/           # chunking (LangChain) e embeddings (EmbeddingGemma via LM Studio)
├── vectorstore/          # Chroma (vector store local, embutido)
├── llm/                  # cliente HTTP para o LM Studio (OpenAI-compatible)
├── retrieval/             # busca top-k fixa e busca adaptativa (margem + teto de contexto)
├── prompts/                # templates de prompt do QA e do resumo (map-reduce)
├── pipeline/                # orquestração: ingest, query e summarize
├── cli/                      # interface de linha de comando (Typer)
└── app/                       # interface web (Streamlit)

scripts/
└── rag-pdf.sh                 # start/stop/restart/status/clean/logs
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
