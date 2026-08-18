# Como o build CUDA do CrispEmbed funciona, por baixo dos panos

Notas de uma conversa tirando dúvidas sobre o que exatamente acontece quando a gente builda o
CrispEmbed com CUDA nesta máquina — que ferramentas fazem o quê, o que tem dentro dos arquivos
`.cu`, e por que uma atualização de driver de GPU não exige recompilar nada (diferente do que
costuma acontecer com shaders de jogos). Complementa o
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) (que é sobre os bugs que apareceram) — este arquivo é
sobre **como o processo funciona quando dá tudo certo**.

## As ferramentas e o papel de cada uma

O repositório do CrispEmbed só tem código-fonte (`.cpp`/`.h`/`.cu`) — nada pré-compilado. O
build precisa gerar código de máquina em **dois níveis diferentes**, com compiladores
diferentes, depois juntar tudo num `.exe` só:

| Ferramenta | Papel |
|---|---|
| **Git** | Clona o código-fonte e baixa o submódulo `ggml` (a biblioteca de tensores por baixo do projeto — mesma "família" usada pelo llama.cpp) |
| **CMake** | Não compila nada sozinho — lê os `CMakeLists.txt` do projeto (o que precisa ser compilado e como) e gera os arquivos de build de verdade pro sistema escolhido |
| **Ninja** | O executor de build — roda as centenas de chamadas de compilação/link em paralelo. Vem embutido no Visual Studio Build Tools, não precisa instalar à parte |
| **MSVC (`cl.exe`)** | Compila o código "hospedeiro" (C++ comum, roda na CPU) e faz a linkagem final dos `.exe` |
| **CUDA Toolkit (`nvcc`)** | Compila especificamente os arquivos `.cu` — os kernels que rodam *dentro* da GPU — e fornece bibliotecas de matemática aceleradas (cuBLAS etc.) |

O "Visual Studio Build Tools" é, na prática, um pacote que já traz três dessas peças juntas
(MSVC, CMake, Ninja); só o CUDA Toolkit é instalado separado, da NVIDIA.

## As etapas do build (`build-cuda.bat`, em ordem)

1. Inicializa o submódulo `ggml` (`git submodule update --init --recursive`), se ainda não
   tiver sido baixado.
2. Localiza o Visual Studio via `vswhere.exe`.
3. Confere se `nvcc` existe no `PATH`.
4. Importa o ambiente do MSVC (`vcvars64.bat`) — configura `INCLUDE`/`LIB`/`PATH` pro `cl.exe`
   achar os headers e bibliotecas do C++ e do Windows SDK.
5. Configura o build com CMake, gerando arquivos Ninja:
   ```
   cmake -G Ninja -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON ...
   ```
6. Compila de fato (`cmake --build build-cuda --config Release`) — o Ninja dispara as chamadas
   ao `cl.exe` (código C++ comum) e ao `nvcc` (kernels `.cu`) em paralelo, e por fim linka tudo
   nos executáveis finais (`crispembed.exe`, `crispembed-server.exe`, etc.).

## O que tem dentro dos arquivos `.cu`

São os **kernels CUDA** — funções escritas numa extensão do C++ que descrevem um cálculo pra
rodar em paralelo, em milhares de threads simultâneas nos núcleos da GPU. No `ggml`, cada
operação matemática tem uma versão CPU (C++ puro) e uma versão GPU (`.cu`): multiplicação de
matriz, atenção, convolução (usada no "patch embedding" do SigLIP), normalização, etc. É a
forma do `ggml`/CrispEmbed acessar a GPU em baixo nível — o `ggml` escolhe qual versão usar em
tempo de execução (por isso o log do servidor mostra `"using CUDA0 backend with CPU
fallback"`).

## Os `.cu` precisam ser compilados pra "a máquina" que vai rodar?

Não pra máquina exata — pra **geração de arquitetura da GPU** (compute capability). Conferido
direto no log do nosso build:

```
Using CMAKE_CUDA_ARCHITECTURES=89-real
```

`89` = compute capability 8.9 = arquitetura **Ada Lovelace**, a família toda das RTX 40 (4090,
4080, 4070...), não só a RTX 4080 desta máquina especificamente. O CMake do `ggml` detecta
automaticamente a GPU presente no momento do build (modo `"native"`).

O sufixo **`-real`** (em vez de `-virtual`) importa: o `nvcc` pode gerar dois tipos de saída:

- **SASS** — código de máquina nativo pra uma arquitetura específica (rápido, só roda naquela
  geração exata).
- **PTX** — um "bytecode" intermediário que o **driver da NVIDIA compila na hora** (JIT) pra
  qualquer GPU sem SASS específico — mais lento na primeira execução, mas portátil.

Como nosso build usou só `-real` (sem `-virtual`), o `.exe` só tem SASS pra 8.9 — nenhum PTX de
fallback embutido:

- Roda liso em qualquer RTX série 40 (mesma arquitetura, 8.9).
- Numa RTX 30 (Ampere, 8.6) ou 20 (Turing, 7.5) não teria código de GPU compatível embutido —
  precisaria recompilar (ou o `ggml` cairia pro fallback de CPU, bem mais lento).

## Atualizar o driver da GPU exige recompilar?

**Não** — e a razão prática já apareceu sozinha no log de startup do servidor:

```
ggml_cuda_init: CUDA minor version mismatch — compiled 13.3, runtime 13.2. This usually works but may cause subtle issues.
```

Compilamos contra o CUDA Toolkit 13.3, o driver instalado só reporta suporte a 13.2 (um degrau
"menor" abaixo) — e mesmo assim rodou de boa, só com aviso. Essa é a garantia de
**compatibilidade de driver** da NVIDIA: um driver igual ou mais novo consegue executar SASS
compilado por uma versão de Toolkit igual ou anterior, sem recompilar nada. O papel do driver
aqui é só *executar* o código de máquina já pronto — ele não recompila.

O que de fato forçaria um rebuild não é atualizar o driver, é trocar a **geração de GPU**
(arquitetura/compute capability diferente) — porque, como vimos acima, esse `.exe` só tem
código de máquina pra 8.9, sem PTX de fallback pra outras gerações.

### Por que isso é diferente de shaders de jogos

Shader de jogo normalmente **não** vem pré-compilado pra arquitetura nenhuma — jogos em PC não
sabem de antemão qual GPU exata o jogador tem. Eles distribuem os shaders num formato
intermediário portátil (DXIL no DirectX, SPIR-V no Vulkan — o equivalente ao PTX mencionado
acima), e é o **driver instalado na máquina do jogador** quem faz a última etapa de compilação
pra código nativo daquela GPU específica — em **tempo de execução**, na primeira vez que aquele
shader é usado. Cada driver mantém um cache desses shaders já compilados; ao **atualizar o
driver**, esse cache costuma ser invalidado (o compilador interno do driver mudou de versão), e
é isso que causa a travadinha ao rejogar tudo de novo depois de atualizar o driver — é o driver
recompilando shader por shader na hora, de novo.

A diferença central: nosso build compilou pra código de máquina **uma vez, aqui, antecipadamente
(ahead-of-time)**, porque sabíamos exatamente qual GPU ia rodar. Um jogo faz basicamente a mesma
compilação, só que **no computador do jogador, na hora** (just-in-time), porque o desenvolvedor
não sabe qual GPU o jogador tem — e cada atualização de driver reseta esse trabalho.

(O CUDA também tem essa via JIT — se o build tivesse incluído PTX em vez de só `-real`, existiria
um cache de JIT parecido, também invalidável por atualização de driver. Só não sofremos isso
porque o `ggml`, por padrão, compila direto pra SASS quando a arquitetura de build é conhecida.)
