# Pipeline do rag-pdf-local

Este documento explica, arquivo por arquivo, o que cada etapa do projeto faz e por quê. A
analogia geral usada ao longo do texto: o sistema funciona como uma **biblioteca com um
bibliotecário-assistente**. A ingestão é o processo de catalogar um livro novo; a consulta é
pedir ajuda ao bibliotecário para responder uma pergunta específica; o resumo é pedir pra ele
ler o livro inteiro e te contar do que se trata.

Há duas pipelines principais — **ingestão** (indexar um PDF) e **consulta/resumo** (responder
perguntas sobre o que foi indexado) — mais uma terceira, menor, de **resumo map-reduce**, e uma
quarta pipeline paralela, para **imagens** (seção 4), que segue a mesma lógica geral mas com um
bibliotecário diferente para cada sentido: um que "olha" a imagem e escreve o que viu, e outro
que tira a impressão digital dessa descrição.

```
                 ┌─────────────────────────── INGESTÃO ───────────────────────────┐
   PDF  ──────▶  pdf_loader  ──▶  cleaner  ──▶  chunker  ──▶  embedder  ──▶  chroma_store
                 (extrai texto)   (limpa)      (corta em     (gera         (indexa no
                                                pedaços)      "impressões   banco vetorial)
                                                              digitais")

                 ┌──────────────────────────── CONSULTA ──────────────────────────┐
Pergunta ─────▶  embedder (query) ──▶ retriever ──▶ qa_prompt ──▶ lmstudio_client ──▶ resposta
                 (impressão          (busca no      (monta o      (pergunta ao
                 digital da          banco por       "dossiê" pro   LLM local)
                 pergunta)           parecença)      LLM ler)

                 ┌─────────────────────────── RESUMO (map-reduce) ────────────────┐
Documento ────▶  chroma_store.get_by_source ──▶ summary_prompt ──▶ lmstudio_client (várias
                 (busca TODOS os                 (map: resume        vezes, em blocos)
                 chunks do doc,                   cada bloco;
                 em ordem de página)               reduce: junta
                                                    os resumos)
```

---

## 1. Ingestão (indexar um PDF)

### `ingestion/pdf_loader.py`

**O que faz:** abre o PDF com a biblioteca `pypdf` e extrai o texto de cada página, uma por
uma. Se uma página não retorna texto nenhum (comum em PDFs escaneados/imagem, sem OCR), ela é
pulada com um aviso no log.

**Analogia:** é o funcionário da biblioteca que folheia o livro físico e datilografa o
conteúdo de cada página numa ficha, numerando a página de origem. Se uma página é só uma
imagem borrada (sem texto selecionável), ele anota "não consegui ler esta página" e segue.

### `ingestion/cleaner.py`

**O que faz:** normaliza o texto extraído — remove caracteres nulos, colapsa espaços/tabs
repetidos, reduz sequências de 3+ quebras de linha para no máximo 2.

**Analogia:** é a revisão que tira borrões e rabiscos acidentais da ficha antes de arquivá-la,
sem reescrever o conteúdo.

### `processing/chunker.py`

**O que faz:** corta o texto de cada página em pedaços menores (`chunks`) de ~700 caracteres
(`CHUNK_SIZE`), com 100 caracteres de sobreposição (`CHUNK_OVERLAP`) entre pedaços vizinhos,
usando o `RecursiveCharacterTextSplitter` do LangChain (que tenta cortar em quebras de
parágrafo/frase antes de cortar no meio de uma palavra). Cada chunk guarda: id único, texto,
nome do arquivo de origem, número da página e índice do chunk dentro da página.

**Por que sobreposição:** se uma ideia importante estiver bem na fronteira entre dois cortes,
a sobreposição garante que ela apareça inteira em pelo menos um dos dois chunks, em vez de
ficar partida ao meio nos dois.

**Analogia:** é transformar cada página do livro em várias fichas de índice menores (tipo
fichário de biblioteca antiga), cada uma com um pedacinho do texto — pra facilitar achar
trechos específicos depois, em vez de ter que reler a página inteira. A sobreposição é como
repetir a última frase de uma ficha no início da próxima, pra não perder o fio da meada.

### `processing/embedder.py`

**O que faz:** para cada chunk de texto, chama o modelo **EmbeddingGemma** (rodando no LM
Studio, via `/v1/embeddings`) e recebe de volta um vetor de números (768 dimensões) que
representa o *significado* daquele texto. Usa prefixos de tarefa diferentes para documentos
(`"title: none | text: ..."`) e para perguntas (`"task: search result | query: ..."`), como
recomendado pelo próprio modelo — isso melhora a qualidade da busca.

**Analogia:** é tirar uma "impressão digital de significado" de cada ficha. Duas fichas que
falam de assuntos parecidos (mesmo com palavras diferentes) ficam com impressões digitais
parecidas — é assim que a busca por similaridade funciona depois, sem precisar bater
palavra-por-palavra.

### `vectorstore/chroma_store.py`

**O que faz:** guarda os chunks (texto + metadados + embedding) no **Chroma**, um banco de
dados vetorial local, embutido (não precisa de servidor separado, é só uma pasta em disco:
`data/vectorstore/`). A coleção é configurada para medir similaridade por **distância de
cosseno** (`hnsw:space: cosine`) — quanto **menor** a distância, mais parecidos os textos.

Cada chunk também carrega um metadado `folder` — uma **pasta virtual** opcional (ex.:
`"Suporte/2024"`), escolhida na hora da ingestão. Não existe pasta de verdade no disco: é só
uma tag hierárquica salva junto do chunk, usada pra organizar e filtrar depois.

Expõe estas operações:
- `add(...)`: insere/atualiza chunks (usa `upsert`, então reindexar o mesmo PDF não duplica)
- `query(embedding, top_k, where=None)`: devolve os `top_k` chunks mais parecidos com um vetor
  de busca; `where` filtra por metadado (ex.: `{"source": {"$in": [...]}}`) pra restringir a
  busca a um subconjunto de documentos
- `get_by_source(source)`: devolve **todos** os chunks de um documento específico, ordenados
  por página — usado pelo resumo, não pela busca
- `list_sources()`: lista os nomes de todos os documentos já indexados
- `list_documents()`: lista `{source, folder}` de cada documento indexado — usado pra montar
  a árvore de pastas na interface
- `list_folders()`: lista as pastas virtuais já usadas
- `count()`: quantos chunks existem no total

> **Analogia:** a pasta virtual é como uma etiqueta colorida grudada na lombada de um livro
> antes de guardá-lo no fichário — não muda onde ele fisicamente fica no acervo (todos os
> chunks continuam na mesma coleção do Chroma), só ajuda a filtrar depois: "me mostre só os
> livros com etiqueta 'Suporte/2024'".

**Analogia:** é o fichário inteligente da biblioteca. Em vez de organizar as fichas em ordem
alfabética, ele organiza por *proximidade de significado* — pedir "me dê as 5 fichas mais
parecidas com esta ideia" é instantâneo, porque as impressões digitais (embeddings) já foram
calculadas e guardadas.

---

## 2. Consulta (perguntar sobre o documento)

Existem **dois jeitos de buscar contexto** para responder uma pergunta, mais um terceiro modo
que ignora a pergunta e lê o documento inteiro (seção 3).

### `retrieval/retriever.py`

Duas estratégias de busca, ambas comparando o embedding da pergunta com os embeddings de
todos os chunks indexados:

**`retrieve()` — top-k fixo.** Pega sempre exatamente `k` chunks: os `k` com menor distância
(mais parecidos), nem um a mais nem a menos.

> **Analogia:** pedir ao bibliotecário "me traga exatamente as 4 fichas mais parecidas com
> isso", mesmo que só 1 seja realmente relevante, ou mesmo que existam 15 igualmente boas.

**`retrieve_adaptive()` — busca adaptativa.** Pega o chunk mais parecido (a melhor
distância) e vai incluindo os próximos, na ordem de similaridade, enquanto:
1. a distância deles estiver dentro de uma **margem** (`margin_pct`) em relação à melhor
   distância encontrada — ex.: margem de 45% permite chunks até 45% "piores" que o melhor; e
2. o total de caracteres acumulado não estourar o **teto de contexto**
   (`max_context_chars`).

Para assim que qualquer um dos dois limites for atingido primeiro. Sempre devolve pelo menos
o melhor chunk, mesmo que ele sozinho já exceda o teto.

> **Analogia:** em vez de "me dê exatamente 4 fichas", é "me dê a melhor ficha, e continue
> trazendo as próximas enquanto elas ainda forem quase tão boas quanto a primeira — mas para
> quando a qualidade cair muito, ou quando minha mochila (o contexto que o LLM vai ler)
> estiver cheia." Se a pergunta tem uma resposta bem concentrada num só lugar do livro, ele
> traz pouco; se está espalhada, ele traz mais.

A busca adaptativa tem **3 perfis prontos** (mais um "Personalizado" com sliders manuais),
definidos em `pipeline/query_pipeline.py::RETRIEVAL_PROFILES`:

| Perfil | Margem | Teto de contexto |
|---|---|---|
| Restrita | 30% | 3.000 caracteres |
| Padrão | 45% | 6.000 caracteres |
| Abrangente | 60% | 14.000 caracteres |

**Escopo por pasta/arquivo:** tanto `retrieve()` quanto `retrieve_adaptive()` aceitam um
parâmetro opcional `sources` (lista de nomes de arquivo). Quando informado, a busca é
restrita só àqueles documentos — traduzido internamente para um filtro `where={"source":
{"$in": sources}}` no Chroma. Na interface, isso é escolhido através de uma árvore de
pastas/arquivos (checkboxes); sem nenhuma seleção, a busca considera todos os documentos
indexados.

### `prompts/qa_prompt.py`

**O que faz:** monta o texto final enviado ao LLM — um `SYSTEM_PROMPT` fixo que instrui o
modelo a responder **só** com base no contexto fornecido (e admitir quando não sabe, citando
a página de origem sempre que possível), e um `build_user_prompt()` que junta os chunks
recuperados (com fonte e página) + a pergunta.

**Analogia:** é o assistente jurídico que organiza um dossiê de provas antes de levar o caso
pro advogado (o LLM) — e avisa o advogado, por regra, "só argumente com base nesses papéis
aqui, não invente".

### `llm/lmstudio_client.py`

**O que faz:** é o cliente HTTP que conversa com o LM Studio (API compatível com OpenAI, em
`http://localhost:1234/v1`), enviando o `SYSTEM_PROMPT` + prompt do usuário e devolvendo a
resposta gerada.

Tem duas proteções específicas, adicionadas depois de problemas reais encontrados em uso:

1. **Detecção de corte por limite de tokens**: modelos "thinking" (que raciocinam antes de
   responder) às vezes gastam todo o orçamento de tokens (`LLM_MAX_TOKENS`) pensando e nunca
   chegam a escrever a resposta final — nesse caso a API retorna `content` vazio com
   `finish_reason=length`. Em vez de devolver uma resposta em branco silenciosamente, o
   cliente levanta um erro explicando o problema.
2. **Limpeza de raciocínio vazado**: alguns modelos, quando a extração de raciocínio está
   desativada no LM Studio, acabam escrevendo o "pensamento" interno diretamente dentro da
   resposta final, às vezes separado por um marcador de canal (`<channel|>`). O cliente
   detecta esse marcador e descarta tudo antes dele, mantendo só a resposta de verdade.

**Analogia:** é o intérprete que liga pro especialista (o LLM local), repassa a pergunta e o
dossiê, e transcreve a resposta — mas foi treinado pra perceber quando o especialista "pensou
alto" antes de responder e cortar esse rascunho da transcrição final.

### `pipeline/query_pipeline.py`

**O que faz:** orquestra a consulta — chama o `retriever` (fixo ou adaptativo), monta o
prompt, chama o LLM, e devolve um `QAResult` (resposta + lista de trechos usados como fonte).
Se não houver nada indexado, devolve uma mensagem pedindo pra rodar o `ingest` primeiro, sem
nem chamar o LLM.

---

## 3. Resumo (map-reduce — ler o documento inteiro)

Pensado para perguntas do tipo *"quais os principais pontos do livro?"*, que a busca por
similaridade (seção 2) não consegue responder bem — porque não existe um trecho "parecido"
com uma pergunta genérica de resumo, então a busca tradicional só traria pedaços aleatórios.

### `pipeline/summarize_pipeline.py`

**O que faz:** ignora completamente o texto da pergunta. Busca **todos** os chunks do
documento escolhido (via `chroma_store.get_by_source`, em ordem de página) e aplica a técnica
de **map-reduce**:

1. **Map:** agrupa os chunks em blocos de até `SUMMARIZE_BATCH_CHARS` caracteres (padrão:
   24.000) e pede ao LLM um resumo de cada bloco, um de cada vez.
2. **Reduce:** se sobrou mais de um resumo parcial, agrupa esses resumos e pede ao LLM pra
   consolidá-los num resumo único — repete esse passo (até 4 níveis) até sobrar só um resumo
   final.

Também expõe um `on_progress` (callback) que é chamado depois de cada chamada ao LLM, com
quantas chamadas já foram feitas, quantas faltam, e o tempo médio por chamada — é isso que
alimenta a barra de progresso com estimativa de tempo restante na interface.

**Analogia:** é dar o livro inteiro pra um grupo de assistentes, cada um lendo e resumindo um
capítulo (map), e depois um editor-chefe junta esses resumos parciais num relatório final
(reduce) — repetindo a rodada de edição se ainda tiver resumo demais pra juntar de uma vez só.

### `prompts/summary_prompt.py`

**O que faz:** define os prompts de sistema e as instruções pro map (resumir um trecho) e
pro reduce (consolidar vários resumos parciais em um só).

---

## 4. Imagens (busca multimodal)

Pipeline paralela à de PDF, pensada para indexar e buscar imagens. A diferença central: em vez
de extrair texto de um documento, um modelo com **visão** "olha" a imagem e escreve uma
descrição — e é essa descrição, mais o embedding visual da imagem em si, que viram as
"impressões digitais" guardadas no fichário.

```
                 ┌──────────────────── INGESTÃO DE IMAGEM ────────────────────┐
   Imagem ────▶  vision_client   ──▶  image_embedder  ──▶  embedder (texto) ──▶  chroma_image_store
                 (descreve a           (SigLIP via            (Gemma sobre a       (2 coleções:
                 imagem em texto,      CrispEmbed/CUDA)        descrição, 768d)     visual 1152d +
                 via LM Studio)                                                    texto 768d)

                 ┌──────────────────────── BUSCA DE IMAGEM ────────────────────────┐
Texto ou imagem ─▶  embedder (texto) ou image_embedder (imagem)  ──▶  chroma_image_store.query
                    (impressão digital da busca)                      (na coleção correspondente)
```

### `llm/vision_client.py`

**O que faz:** envia a imagem (como base64, no formato `image_url` da API de chat compatível
com OpenAI) para um modelo **com suporte a visão** carregado no LM Studio, junto de um prompt
(`prompts/vision_prompt.py`) pedindo uma descrição completa e precisa — objetos, texto visível,
cores, composição, contexto. Devolve o texto gerado.

**Por que a descrição precisa ser completa:** é esse texto que alimenta a busca por palavra-chave
depois (via embedding de texto) — uma descrição vaga ("uma foto") não ajuda ninguém a encontrar
a imagem de novo; uma descrição detalhada sim.

**Analogia:** é o funcionário da biblioteca que olha para uma foto e dita, em voz alta, tudo que
vê nela, para outro funcionário catalogar depois.

### `processing/image_embedder.py`

**O que faz:** gera o embedding **visual** da imagem (1152 dimensões) usando o modelo **SigLIP**
em formato GGUF (`siglip-so400m-patch14-384`). Como o LM Studio ainda não suporta esse modelo,
ele roda **localmente via CUDA** através do [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)
— um runtime local, com build CUDA, exposto como servidor HTTP (mesmo padrão do LM Studio: um
processo local, um cliente HTTP fino conversando com ele).

**Analogia:** é uma segunda impressão digital, tirada não do que foi *dito* sobre a imagem, mas
da imagem *em si* — captura semelhanças visuais (cores, formas, composição) que uma descrição em
palavras poderia não capturar.

### `vectorstore/chroma_image_store.py`

**O que faz:** guarda cada imagem em **duas coleções Chroma** — uma para o embedding visual
(SigLIP, 1152d) e outra para o embedding da descrição (Gemma, 768d) — porque uma coleção Chroma
só suporta uma dimensão de vetor por vez. As duas coleções compartilham o mesmo `id` (o nome do
arquivo) e os mesmos metadados (caminho da imagem em disco, descrição, dimensões, pasta), então
um resultado de qualquer uma das duas dá acesso à imagem completa.

**Analogia:** é o mesmo fichário inteligente da seção 1, só que com duas gavetas paralelas para
cada imagem — uma organizada por "o que ela parece", outra por "o que foi dito sobre ela" — e
uma etiqueta comum ligando as duas gavetas.

### `pipeline/image_ingest_pipeline.py`

**O que faz:** orquestra a ingestão — lê dimensões/formato da imagem, chama `vision_client` pra
gerar a descrição, `image_embedder` pra gerar o embedding visual, `processing/embedder.py`
(o mesmo usado pelos PDFs) pra gerar o embedding da descrição, e salva tudo no
`chroma_image_store`.

### `pipeline/image_query_pipeline.py`

**O que faz:** expõe três modos de busca, escolhidos pelo usuário na interface:

- **`run_image_text_query()`** — busca por texto: embute a pergunta com o Gemma (mesmo
  `embed_query` usado nos PDFs) e compara com os embeddings de descrição.
- **`run_image_siglip_text_query()`** — busca por texto nativa do SigLIP: embute o texto com a
  **torre de texto** do próprio SigLIP (`processing/image_embedder.py::embed_text_siglip()`,
  via CrispEmbed `POST /clip/text`) e compara **direto** com os embeddings visuais — sem passar
  pela descrição do modelo de visão. Precisa de um `.gguf` separado da torre de texto (gerado
  com o conversor do próprio CrispEmbed a partir do checkpoint original no Hugging Face — veja
  [embedding_image_server/README.md](../embedding_image_server/README.md)), carregado junto do
  servidor via `--clip-text`. As distâncias ficam mais compactadas (perto de 1.0) que nos outros
  modos — característica de como o SigLIP foi treinado (perda sigmoid com escala/bias
  aprendidos, não aplicados aqui), não indica busca pior; o que importa é a ordem relativa.
- **`run_image_similarity_query()`** — busca por imagem parecida: embute uma imagem de exemplo
  com o SigLIP e compara com os embeddings visuais das imagens já indexadas.

> **Analogia:** é pedir ao bibliotecário "me traga fotos parecidas com essa descrição" (busca
> por texto via Gemma), "me traga fotos que combinam com essa palavra" (busca nativa do
> SigLIP — mais direta, menos "conversada") ou "me traga fotos parecidas com essa aqui que eu
> tenho na mão" (busca por imagem) — três jeitos diferentes de perguntar coisas parecidas, cada
> um melhor numa situação.

## 5. Interfaces

### `cli/main.py`

**O que faz:** expõe três comandos via Typer:
- `rag-pdf ingest --pdf <arquivo>` → chama `ingest_pipeline.run_ingest`
- `rag-pdf ask "<pergunta>"` → chama `query_pipeline.run_query` (busca top-k fixa)
- `rag-pdf summarize [--source <arquivo>]` → chama `summarize_pipeline.run_summarize`, com
  progresso impresso no terminal

### `app/streamlit_app.py` + `app/image_tab.py`

**O que faz:** interface web (Streamlit) com duas abas:
- **📄 PDFs** (todo o corpo original do app): **barra lateral** com upload/indexação de PDF,
  contador de chunks no vector store, e o seletor de **modo de resposta** (top-k fixo / busca
  adaptativa com os 3 perfis + manual / resumo map-reduce); **chat** com histórico de perguntas
  e respostas, fontes exibidas num expander (arquivo, página, distância); **resumo** — quando
  esse modo está ativo, qualquer mensagem enviada dispara o map-reduce no documento selecionado,
  com barra de progresso e tempo estimado ao vivo.
- **🖼️ Imagens** (`app/image_tab.py`, chamada de dentro da aba): upload de imagens com
  indexação (mostra a descrição gerada), seletor de busca **por texto** ou **por imagem
  parecida**, e uma galeria das imagens já indexadas com exclusão individual.

`app/ui_helpers.py` guarda o `confirm_icon_action` (botão de exclusão com confirmação em duas
etapas), reaproveitado pelas duas abas.

### `config.py`

**O que faz:** centraliza todas as configurações ajustáveis via `.env` (usando
`pydantic-settings`) — tamanho de chunk, modelo de embedding, caminho do vector store, `top_k`
padrão, URL/modelo do LM Studio, temperatura, limite de tokens, tamanho de bloco do resumo.

**Analogia:** é o painel de controle do sistema — todos os "botões" ajustáveis num só lugar,
com valores padrão sensatos que podem ser sobrescritos sem mexer em código.

---

## Resumo do fluxo ponta a ponta

1. Você sobe um PDF → `pdf_loader` extrai texto → `cleaner` limpa → `chunker` corta em
   pedaços → `embedder` gera a "impressão digital" de cada pedaço → `chroma_store` guarda
   tudo no banco vetorial local.
2. Você faz uma pergunta → `embedder` gera a impressão digital da pergunta → `retriever`
   busca os pedaços mais parecidos (fixo ou adaptativo) → `qa_prompt` monta o dossiê →
   `lmstudio_client` pergunta pro LLM local → você recebe a resposta com as fontes.
3. Você pede um resumo → `summarize_pipeline` busca o documento inteiro, ignorando a
   pergunta → resume em blocos (map) → consolida os resumos (reduce) → você recebe o resumo
   final, com barra de progresso ao longo do processo.
4. Você sobe uma imagem → `vision_client` gera uma descrição via modelo de visão no LM Studio →
   `image_embedder` gera o embedding visual (SigLIP, via CrispEmbed/CUDA) → `embedder` gera o
   embedding da descrição (Gemma) → `chroma_image_store` guarda os dois. Depois, você busca por
   texto (compara com o embedding da descrição) ou por imagem parecida (compara com o embedding
   visual) → recebe as imagens mais parecidas, com a descrição gerada para cada uma.
