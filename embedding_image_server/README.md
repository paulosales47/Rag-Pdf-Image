# embedding_image_server

Build local do [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed) com aceleração **CUDA**,
usado só para servir o embedding visual **SigLIP** (`siglip-so400m-patch14-384.gguf`) — o LM
Studio ainda não suporta esse modelo, então ele roda por fora, como um servidor HTTP local
próprio (mesma ideia do LM Studio: um processo local, um cliente HTTP fino conversando com ele
a partir do `rag_pdf`, em `src/rag_pdf/processing/image_embedder.py`).

O código-fonte do CrispEmbed **não é versionado neste repositório** — `setup.ps1` clona e builda
localmente dentro desta pasta (em `CrispEmbed/`, ignorado pelo git).

## Pré-requisitos

Ferramentas de build que precisam estar instaladas e no `PATH` **antes** de rodar `setup.ps1`:

- **Git**
- **CMake** (>= 3.20)
- **Visual Studio 2022 Build Tools**, com o workload "Desktop development with C++" (fornece o
  compilador `cl.exe` usado pelo CMake no Windows)
- **CUDA Toolkit** (versão compatível com o driver da GPU — confira com `nvidia-smi`; a máquina
  atual reporta CUDA 13.2 no driver, então o Toolkit 12.x ou 13.x deve funcionar)

`setup.ps1` localiza e importa o ambiente do Visual Studio Build Tools automaticamente (mesmo
rodando de um PowerShell comum, não precisa ser o "Developer PowerShell") e ajusta o `PATH`
desta sessão pro CUDA Toolkit se necessário — só precisa que as três ferramentas estejam
**instaladas**, não necessariamente já no `PATH`.

## Uso

```powershell
# 1. Builda o CrispEmbed com CUDA (só precisa rodar uma vez, ou de novo pra atualizar)
.\embedding_image_server\setup.ps1

# 2. Baixe o modelo (ver instruções abaixo) e coloque em:
#    models\siglip-so400m-patch14-384.gguf

# 3. Suba o servidor (porta 8081 por padrão, igual ao CRISPEMBED_BASE_URL do .env)
.\embedding_image_server\start-server.ps1
```

`start-server.ps1` localiza automaticamente o executável gerado pelo build (não é garantido que
o CMake sempre produza o binário no mesmo caminho relativo em toda versão do CrispEmbed) e o
mantém rodando em primeiro plano — feche a janela ou `Ctrl+C` pra parar.

## Baixando o modelo SigLIP (torre de visão)

Baixe `siglip-so400m-patch14-384.gguf` do repositório
[`cstr/siglip-so400m-patch14-384-GGUF`](https://huggingface.co/cstr/siglip-so400m-patch14-384-GGUF)
no Hugging Face e salve em `models\siglip-so400m-patch14-384.gguf` (pasta na raiz do projeto,
irmã de `src/` — não dentro de `embedding_image_server/`).

## Gerando o modelo de texto (opcional — busca por texto nativa do SigLIP)

O `.gguf` acima só tem a torre de **visão** — o SigLIP original também tem uma torre de
**texto**, que não vem pronta em nenhum repositório GGUF conhecido. Pra habilitar o terceiro
modo de busca da aba de Imagens ("Por texto (nativo do SigLIP)"), gere localmente a partir do
checkpoint original no Hugging Face
([`google/siglip-so400m-patch14-384`](https://huggingface.co/google/siglip-so400m-patch14-384)):

```powershell
# Dependências pra rodar o conversor (só precisa uma vez; baixa ~200MB de torch CPU-only)
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install transformers gguf sentencepiece

# Gera o .gguf de texto (baixa o checkpoint original do HF, alguns GB, na primeira vez)
.\.venv\Scripts\python.exe embedding_image_server\CrispEmbed\models\convert-clip-text-to-gguf.py `
    --model google/siglip-so400m-patch14-384 `
    --output models\siglip-so400m-patch14-384-text.gguf
```

`start-server.ps1` já detecta esse arquivo automaticamente (se existir em `models/`) e carrega
os dois modelos juntos (`--vit` + `--clip-text`), habilitando `POST /clip/text` ao lado de
`POST /vit/encode`. Sem o arquivo, o servidor sobe normalmente só com a busca por imagem.

> Se a conversão falhar com erro de SSL (`CERTIFICATE_VERIFY_FAILED`) tentando conectar no
> Hugging Face, é sintoma comum de antivírus fazendo inspeção de HTTPS com certificado próprio
> — rode `pip install pip-system-certs` no venv (usa o repositório de certificados do Windows
> em vez do bundle padrão do Python) e tente de novo.

## Documentação adicional

- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — cada bug encontrado durante o build/execução, com
  causa raiz, como foi descoberto e a correção aplicada.
- [CUDA_BUILD_EXPLAINED.md](CUDA_BUILD_EXPLAINED.md) — como o build CUDA funciona por baixo dos
  panos (ferramentas, etapas, o que são os arquivos `.cu`, por que a arquitetura da GPU importa
  mas o driver não exige recompilar).

## Testado de ponta a ponta

Build (MSVC 2026 + CUDA 13.3, RTX 4080), subida do servidor e uma chamada real a
`POST /vit/encode` foram validados nesta máquina. Detalhes que valem registrar pra quem rodar
em outra máquina/versão:

- **SigLIP no CrispEmbed é um "ViT standalone"**, carregado com a flag `--vit <arquivo.gguf>`
  — **não** `-m` (essa flag é pro modelo de texto/BERT do CrispEmbed e falha com
  `missing token_embd.weight` se apontada pra um encoder de imagem puro). `start-server.ps1` já
  usa `--vit` corretamente.
- O endpoint correto é `POST /vit/encode` (não `/image`), request
  `{"image": "<caminho absoluto no disco do servidor>"}`, resposta
  `{"embedding": [...], "dim": N}` — confirmado lendo `examples/server/server.cpp` do
  CrispEmbed. `image_embedder.py` já usa esse formato.
- **Bug conhecido do `vswhere`**: o `build-cuda.bat` do próprio CrispEmbed chama `vswhere`
  sem `-products *`, o que ignora a SKU "Build Tools" (só enxerga Community/Professional/
  Enterprise por padrão) e falha com "Visual Studio C++ build tools not found" mesmo com tudo
  instalado. `setup.ps1` aplica esse patch automaticamente na cópia clonada localmente (não
  afeta o repositório upstream).
- **CUDA Toolkit 13.x** move as DLLs de runtime (`cublas64_13.dll`, `cudart64_13.dll` etc.) pra
  `bin\x64\`, não direto em `bin\` como em versões mais antigas — sem isso no `PATH`, o
  executável falha instantaneamente e sem mensagem nenhuma (`STATUS_DLL_NOT_FOUND`). Os dois
  scripts já incluem essa subpasta.
- **Busca por texto nativa do SigLIP** (`--clip-text`, `POST /clip/text`) também validada: uma
  imagem de moto indexada ficou em 1º lugar buscando por `"moto"` e por
  `"uma foto de uma motocicleta"`, à frente de outras 3 imagens não relacionadas — confirma que
  a torre de texto convertida (`convert-clip-text-to-gguf.py`) cai no mesmo espaço vetorial da
  torre de visão. As distâncias de cosseno ficam compactadas perto de 1.0 (bem diferentes dos
  valores da busca via descrição do modelo de visão) — característica do SigLIP (perda sigmoid,
  sem a escala/bias aprendidos aplicados aqui), não indica busca ruim; o que importa é a ordem
  relativa dos resultados.
